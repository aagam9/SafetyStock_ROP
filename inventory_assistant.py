"""Question planning, grounded evidence construction, and Nemotron integration.

The relationship-aware calculations live in :mod:`inventory_data`. This module
interprets a question, selects approved data sources/operations, builds a small
evidence package, and asks Nemotron to explain Python-generated facts.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from inventory_data import InventoryDataError, InventoryDataModel


NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")
LOCAL_ENV_PATH = Path(__file__).resolve().parent / ".env"
MAX_HISTORY_MESSAGES = 6
MAX_CONTEXT_ROWS = 15
SESSION_ANALYSIS_KEYS = (
    "last_report", "last_summary", "item_master", "demand_history",
    "receipt_history", "analysis_results", "review_queue", "analysis_summary",
    "inventory_chat", "assistant_metadata", "assistant_data_model",
    "last_input_signature",
)

SYSTEM_PROMPT = """You are an inventory analytics assistant. Use only the explicit
Python-generated evidence in the current request. Do not invent SKUs, suppliers,
demand, stockouts, lead times, calculations, dates, or business events. Do not
reuse a prior assistant sentence as data when current evidence is supplied.
Distinguish facts from interpretation. Stockout Risk is a relative planning
indicator, not historical stockout evidence. Supplier evidence represents the
current primary-supplier mapping and lead-time behavior, not OTIF, fill rate,
quality, or historical PO-level supplier identity. Answer directly, mention the
important supplied numbers, state material limitations, and stay concise. The
application adds source provenance separately, so do not fabricate citations."""


class InventoryAssistantError(RuntimeError):
    """Safe, user-facing error from assistant configuration or execution."""


class MissingNvidiaAPIKey(InventoryAssistantError):
    """Raised when only the optional assistant is missing its credential."""


@dataclass
class AssistantAnswer:
    text: str
    table: pd.DataFrame | None
    metadata: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionPlan:
    intent: str
    skus: list[str]
    suppliers: list[str]
    time_window: int | None
    required_sources: list[str]
    operations: list[str]
    filters: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    ambiguous_entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reset_inventory_session(state: Any) -> None:
    """Invalidate raw data, derived views, and chat together on input changes."""
    for key in SESSION_ANALYSIS_KEYS:
        state.pop(key, None)


def _read_local_env(path: str | Path | None = None) -> dict[str, str]:
    """Read local development configuration without logging or mutating values."""
    env_path = Path(path) if path is not None else LOCAL_ENV_PATH
    if not env_path.is_file():
        return {}
    try:
        lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", stripped)
        if not match:
            continue
        name, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[name] = value
    return values


def get_nvidia_config(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, str]:
    local_env = _read_local_env()
    resolved_key = (
        api_key or os.getenv("NVIDIA_API_KEY") or local_env.get("NVIDIA_API_KEY", "")
    ).strip()
    if not resolved_key:
        raise MissingNvidiaAPIKey(
            "Inventory analysis is available, but the AI assistant requires an NVIDIA API key."
        )
    return {
        "api_key": resolved_key,
        "model": (
            model or os.getenv("NVIDIA_MODEL") or local_env.get("NVIDIA_MODEL") or NVIDIA_MODEL
        ).strip(),
        "base_url": (
            base_url or os.getenv("NVIDIA_BASE_URL")
            or local_env.get("NVIDIA_BASE_URL") or NVIDIA_BASE_URL
        ).rstrip("/"),
    }


def _results_frame(analysis_results: Sequence[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    frame = (
        analysis_results.copy()
        if isinstance(analysis_results, pd.DataFrame)
        else pd.DataFrame(list(analysis_results))
    )
    if frame.empty:
        raise ValueError("Analysis results are unavailable. Run the inventory analysis first.")
    return frame


# Backward-compatible analytical helpers used by existing callers/tests.
def get_sku_summary(
    analysis_results: Sequence[Mapping[str, Any]] | pd.DataFrame, sku: str
) -> dict[str, Any] | None:
    frame = _results_frame(analysis_results)
    match = frame.loc[frame["sku"].astype(str).str.casefold() == str(sku).strip().casefold()]
    return None if match.empty else match.iloc[0].to_dict()


def forecast_demand(demand_history: pd.DataFrame, sku: str, horizon: int = 4) -> dict[str, Any]:
    """Deterministic 13-period damped linear-trend forecast."""
    horizon = int(horizon)
    if not 1 <= horizon <= 26:
        raise ValueError("Forecast horizon must be between 1 and 26 periods.")
    rows = demand_history.loc[
        demand_history["sku"].astype(str).str.casefold() == str(sku).strip().casefold(),
        "demand_qty",
    ]
    values = pd.to_numeric(rows, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 3:
        return {
            "sku": sku,
            "available_observations": int(len(values)),
            "limitation": "At least 3 demand observations are required for a forecast.",
            "forecast": pd.DataFrame(),
        }
    recent = values[-min(13, len(values)) :]
    x = np.arange(len(recent), dtype=float)
    slope, intercept = np.polyfit(x, recent, 1)
    fitted_latest = intercept + slope * (len(recent) - 1)
    future = np.arange(1, horizon + 1, dtype=float)
    prediction = np.maximum(0.0, fitted_latest + 0.5 * slope * future)
    return {
        "sku": sku,
        "horizon": horizon,
        "method": "13-period linear trend with 50% trend damping",
        "available_observations": int(len(values)),
        "observations_used": int(len(recent)),
        "recent_average": round(float(recent.mean()), 1),
        "recent_trend_units_per_period": round(float(slope), 2),
        "forecast_total": round(float(prediction.sum()), 1),
        "forecast": pd.DataFrame({
            "sku": [sku] * horizon,
            "forecast_period": np.arange(1, horizon + 1),
            "forecast_units": np.round(prediction, 1),
        }),
        "limitation": "Forecast is a planning estimate based only on uploaded demand history.",
    }


def rank_safety_stock_gaps(
    analysis_results: Sequence[Mapping[str, Any]] | pd.DataFrame,
    limit: int = 10,
    direction: str = "increase",
    item_class: str | None = None,
    supplier: str | None = None,
) -> pd.DataFrame:
    frame = _results_frame(analysis_results).copy()
    frame["safety_stock_gap"] = frame["recommended_safety_stock"] - frame["current_safety_stock"]
    if item_class:
        frame = frame.loc[frame["item_class"].astype(str).str.upper() == item_class.upper()]
    if supplier:
        frame = frame.loc[frame["supplier"].astype(str).str.casefold() == supplier.casefold()]
    ascending = direction == "decrease"
    frame = frame.loc[frame["safety_stock_gap"] < 0 if ascending else frame["safety_stock_gap"] > 0]
    wanted = [
        "sku", "description", "supplier", "item_class", "current_safety_stock",
        "recommended_safety_stock", "safety_stock_gap", "recent_weekly_std",
        "lt_actual_std_days",
    ]
    return frame.sort_values("safety_stock_gap", ascending=ascending).head(limit)[
        [column for column in wanted if column in frame.columns]
    ]


def rank_rop_gaps(
    analysis_results: Sequence[Mapping[str, Any]] | pd.DataFrame,
    limit: int = 10,
    direction: str = "absolute",
    supplier: str | None = None,
) -> pd.DataFrame:
    frame = _results_frame(analysis_results).copy()
    frame["rop_gap"] = frame["recommended_rop"] - frame["current_rop"]
    if supplier:
        frame = frame.loc[frame["supplier"].astype(str).str.casefold() == supplier.casefold()]
    if direction == "increase":
        frame = frame.loc[frame["rop_gap"] > 0].sort_values("rop_gap", ascending=False)
    elif direction == "decrease":
        frame = frame.loc[frame["rop_gap"] < 0].sort_values("rop_gap")
    else:
        frame = frame.assign(_magnitude=frame["rop_gap"].abs()).sort_values(
            "_magnitude", ascending=False
        )
    wanted = ["sku", "description", "supplier", "item_class", "current_rop", "recommended_rop", "rop_gap"]
    return frame.head(limit)[wanted]


def calculate_stockout_risk(
    analysis_results: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Legacy-compatible relative score using fields in analysis_results."""
    frame = _results_frame(analysis_results).copy()
    rec_ss = frame["recommended_safety_stock"].replace(0, np.nan)
    rec_rop = frame["recommended_rop"].replace(0, np.nan)
    mean_demand = frame["recent_weekly_mean"].replace(0, np.nan)
    mean_lt = frame["lt_actual_mean_days"].replace(0, np.nan)
    ss_gap = ((frame["recommended_safety_stock"] - frame["current_safety_stock"]) / rec_ss).clip(0, 1)
    rop_gap = ((frame["recommended_rop"] - frame["current_rop"]) / rec_rop).clip(0, 1)
    demand_cv = (frame["recent_weekly_std"] / mean_demand).clip(0, 2) / 2
    lead_cv = (frame["lt_actual_std_days"] / mean_lt).clip(0, 1)
    trend = frame["trend_pct"].clip(lower=0, upper=100) / 100
    criticality = frame["item_class"].map({"A": 1.0, "B": 0.6, "C": 0.3}).fillna(0.6)
    frame["safety_stock_gap"] = frame["recommended_safety_stock"] - frame["current_safety_stock"]
    frame["rop_gap"] = frame["recommended_rop"] - frame["current_rop"]
    frame["stockout_risk_score"] = (100 * (
        0.30 * ss_gap.fillna(0) + 0.20 * rop_gap.fillna(0)
        + 0.15 * demand_cv.fillna(0) + 0.15 * lead_cv.fillna(0)
        + 0.10 * trend.fillna(0) + 0.10 * criticality
    )).round(1)
    frame["stockout_risk_level"] = pd.cut(
        frame["stockout_risk_score"], [-np.inf, 25, 50, np.inf], labels=["Low", "Moderate", "High"]
    ).astype(str)
    columns = [
        "sku", "description", "supplier", "item_class", "stockout_risk_score",
        "stockout_risk_level", "safety_stock_gap", "rop_gap", "trend_pct",
        "recent_weekly_std", "lt_actual_std_days",
    ]
    return frame.sort_values("stockout_risk_score", ascending=False)[columns]


def get_supplier_summary(
    item_master: pd.DataFrame,
    receipt_history: pd.DataFrame,
    analysis_results: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Compatibility supplier rollup; receipt and result tables join only after aggregation."""
    receipt_supplier = receipt_history.merge(
        item_master[["sku", "supplier"]], on="sku", how="left", validate="many_to_one"
    )
    receipt_stats = receipt_supplier.groupby("supplier", dropna=False).agg(
        sku_count=("sku", "nunique"), receipt_count=("actual_lead_time_days", "size"),
        average_lead_time_days=("actual_lead_time_days", "mean"),
        median_lead_time_days=("actual_lead_time_days", "median"),
        minimum_lead_time_days=("actual_lead_time_days", "min"),
        maximum_lead_time_days=("actual_lead_time_days", "max"),
        lead_time_std_days=("actual_lead_time_days", "std"),
    )
    receipt_stats["lead_time_std_days"] = receipt_stats["lead_time_std_days"].fillna(0)
    receipt_stats["lead_time_cv"] = receipt_stats["lead_time_std_days"] / receipt_stats[
        "average_lead_time_days"
    ].replace(0, np.nan)
    results = _results_frame(analysis_results).copy()
    results["safety_stock_gap"] = results["recommended_safety_stock"] - results["current_safety_stock"]
    results["rop_gap"] = results["recommended_rop"] - results["current_rop"]
    risk = calculate_stockout_risk(results)[["sku", "stockout_risk_score"]]
    results = results.merge(risk, on="sku", how="left", validate="one_to_one")
    planning = results.groupby("supplier", dropna=False).agg(
        total_current_safety_stock=("current_safety_stock", "sum"),
        total_recommended_safety_stock=("recommended_safety_stock", "sum"),
        total_safety_stock_gap=("safety_stock_gap", "sum"),
        total_rop_gap=("rop_gap", "sum"),
        average_stockout_risk_score=("stockout_risk_score", "mean"),
        average_demand_variability=("recent_weekly_std", "mean"),
    )
    summary = receipt_stats.join(planning, how="outer").reset_index()
    numeric = summary.select_dtypes(include="number").columns
    summary[numeric] = summary[numeric].round(2)
    return summary.sort_values("average_lead_time_days", ascending=False)


def compare_skus(
    analysis_results: Sequence[Mapping[str, Any]] | pd.DataFrame, skus: Sequence[str]
) -> pd.DataFrame:
    frame = _results_frame(analysis_results).copy()
    wanted = {sku.casefold() for sku in skus}
    frame = frame.loc[frame["sku"].astype(str).str.casefold().isin(wanted)]
    columns = [
        "sku", "description", "supplier", "item_class", "recent_weekly_mean",
        "recent_weekly_std", "trend_pct", "zero_demand_share", "is_intermittent",
        "has_outliers", "assumed_lead_time_days", "lt_actual_mean_days",
        "lt_actual_std_days", "lt_drift_pct", "service_level_z",
        "current_safety_stock", "recommended_safety_stock", "current_rop",
        "recommended_rop", "ss_drift_pct", "analysis_flags",
    ]
    return frame[[column for column in columns if column in frame.columns]]


def compare_suppliers(supplier_summary: pd.DataFrame, suppliers: Sequence[str]) -> pd.DataFrame:
    wanted = {supplier.casefold() for supplier in suppliers}
    return supplier_summary.loc[
        supplier_summary["supplier"].astype(str).str.casefold().isin(wanted)
    ].copy()


def _normalize_entity(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _boundary_match(question: str, value: str) -> bool:
    return bool(re.search(
        rf"(?<![a-z0-9]){re.escape(str(value).casefold())}(?![a-z0-9])",
        question.casefold(),
    ))


def _resolve_entities(
    question: str, item_master: pd.DataFrame, prior_metadata: Mapping[str, Any] | None
) -> tuple[list[str], list[str], list[str]]:
    q = question.casefold()
    normalized_q = _normalize_entity(question)
    skus = [str(value) for value in item_master["sku"].dropna().unique() if _boundary_match(q, str(value))]
    suppliers = [
        str(value) for value in item_master["supplier"].dropna().unique()
        if _boundary_match(q, str(value))
    ]
    descriptions = item_master[["sku", "description"]].dropna()
    for _, row in descriptions.iterrows():
        description = str(row["description"])
        if len(description) >= 4 and _boundary_match(q, description):
            skus.append(str(row["sku"]))

    # Normalized SKU matching handles punctuation/case differences.
    if not skus:
        normalized_matches = [
            str(value) for value in item_master["sku"].dropna().unique()
            if len(_normalize_entity(str(value))) >= 3
            and _normalize_entity(str(value)) in normalized_q
        ]
        skus.extend(normalized_matches)

    ambiguous: list[str] = []
    if not suppliers:
        alias_matches: list[str] = []
        for value in item_master["supplier"].dropna().unique():
            supplier = str(value)
            alias = re.sub(r"^(supplier|vendor)\s+", "", supplier, flags=re.I).strip()
            if len(alias) >= 3 and _boundary_match(q, alias):
                alias_matches.append(supplier)
        if len(set(alias_matches)) == 1:
            suppliers.extend(alias_matches)
        elif len(set(alias_matches)) > 1:
            ambiguous.append("supplier: " + ", ".join(sorted(set(alias_matches))))

    is_follow_up = bool(re.fullmatch(
        r"\s*(why\??|explain|tell me more|what about (it|its .+)|and (it|that one)\??)\s*",
        q,
    ))
    prior = dict(prior_metadata or {})
    resolved_follow_up_sku = False
    if is_follow_up and not skus:
        focus_sku = prior.get("focus_sku")
        if focus_sku:
            skus = [str(focus_sku)]
            resolved_follow_up_sku = True
        else:
            skus = [str(value) for value in prior.get("skus", [])[:1]]
            resolved_follow_up_sku = bool(skus)
    # A ranked result may carry both SKU and supplier metadata. Pronouns such as
    # "it" and a bare "Why?" refer to the focused SKU first; only fall back to
    # supplier context when there is no focused SKU.
    if is_follow_up and not suppliers and not resolved_follow_up_sku and not skus:
        focus_supplier = prior.get("focus_supplier")
        if focus_supplier:
            suppliers = [str(focus_supplier)]
        else:
            suppliers = [str(value) for value in prior.get("suppliers", [])[:1]]
    return sorted(set(skus)), sorted(set(suppliers)), ambiguous


def _time_window(question: str) -> int | None:
    q = question.casefold()
    match = re.search(r"\b(?:last|past|recent)\s+(\d{1,2})\s+(?:weeks?|periods?)\b", q)
    if match:
        return min(max(int(match.group(1)), 1), 52)
    if "last quarter" in q or "recent quarter" in q:
        return 13
    if "recent" in q:
        return 8
    return None


def _limit_from_question(question: str, default: int = 10) -> int:
    match = re.search(r"\btop\s+(\d{1,2})\b", question.casefold())
    return min(max(int(match.group(1)), 1), MAX_CONTEXT_ROWS) if match else default


def plan_question(
    question: str,
    item_master: pd.DataFrame,
    prior_metadata: Mapping[str, Any] | None = None,
) -> QuestionPlan:
    """Create a validated plan selecting only approved sources and operations."""
    q = question.casefold()
    skus, suppliers, ambiguous = _resolve_entities(question, item_master, prior_metadata)
    window = _time_window(question)
    explicit_sku = re.search(r"\bsku\s+([a-z0-9][a-z0-9._/-]*)", question, re.I)
    if explicit_sku and not skus:
        unknown = explicit_sku.group(1).rstrip("?.!,")
        return QuestionPlan(
            "unknown_entity", [], suppliers, window, ["item_master"],
            ["resolve_sku"], limitations=[f"Unknown SKU: {unknown}"],
        )
    unsupported = re.search(r"\b(otif|fill rate|quality|promised[- ]date|on[- ]time delivery)\b", q)
    historical_stockout = (
        ("stockout" in q or "stock out" in q)
        and re.search(r"\b(how many|history|historical|last year|occurred|events?)\b", q)
        and "risk" not in q
    )
    if unsupported or historical_stockout:
        return QuestionPlan("unsupported", skus, suppliers, window, [], ["report_limitation"], ambiguous_entities=ambiguous)

    # Generic "variability" may describe receipt lead times, so demand-oriented
    # questions must use an explicit demand/forecast term (or a demand trend term).
    has_demand = bool(re.search(r"demand|forecast|intermittent|growing|rising|declining", q))
    has_receipt = bool(re.search(r"receipt|shipment|lead[ -]?time|purchase order|\bpo\b|supply", q))
    has_ss = bool(re.search(r"safety stock|\bss\b|under[- ]protected|over[- ]protected", q))
    has_rop = bool(re.search(r"reorder point|\brop\b", q))
    has_risk = bool(re.search(r"stockout|risk|exposed", q))
    is_compare = "compare" in q or " versus " in q or " vs " in q
    asks_about_suppliers = bool(re.search(r"\bsuppliers?\b|\bvendors?\b", q))

    if re.search(r"forecast|predict|projection|next \d+ (?:weeks?|periods?)", q):
        intent = "forecast"
        sources = ["demand_history"]
        operations = ["resolve_sku", "fit_deterministic_forecast"]
    elif skus and re.search(r"what supplier|which supplier|who supplies|provides?\s+(?:sku\s+)?|belong", q):
        intent = "supplier_lookup"
        sources = ["item_master"]
        operations = ["lookup_item_master"]
    elif is_compare and len(suppliers) >= 2:
        intent = "supplier_comparison"
        sources = ["item_master", "demand_history", "receipt_history", "analysis_results"]
        operations = ["resolve_supplier_skus", "aggregate_each_fact_by_sku", "compare_supplier_profiles"]
    elif is_compare and len(skus) >= 2:
        intent = "sku_comparison"
        sources = ["item_master", "demand_history", "receipt_history", "analysis_results"]
        operations = ["build_sku_profiles", "compare_profiles"]
    elif asks_about_suppliers and not suppliers and (has_ss or has_rop or has_risk):
        intent = "supplier_overview"
        sources = ["item_master", "demand_history", "receipt_history", "analysis_results"]
        operations = ["build_supplier_profiles", "rank_suppliers"]
    elif has_ss and has_demand and not skus:
        intent = "cross_file_risk"
        sources = ["item_master", "demand_history", "receipt_history", "analysis_results"]
        operations = ["aggregate_demand_by_sku", "aggregate_receipts_by_sku", "combine_at_sku_grain", "filter_and_rank"]
    elif has_ss and suppliers:
        intent = "supplier_inventory"
        sources = ["item_master", "demand_history", "receipt_history", "analysis_results"]
        operations = ["resolve_supplier_skus", "aggregate_each_fact_by_sku", "rank_safety_stock_gaps"]
    elif has_ss:
        intent = "sku_safety_stock" if skus else "inventory_ranking"
        sources = ["item_master", "demand_history", "receipt_history", "analysis_results"]
        operations = ["build_sku_profile", "explain_safety_stock_drivers"] if skus else ["rank_safety_stock_gaps"]
    elif has_rop:
        intent = "sku_rop" if skus else "inventory_ranking"
        sources = ["item_master", "demand_history", "receipt_history", "analysis_results"]
        operations = ["build_sku_profile", "explain_rop_drivers"] if skus else ["rank_rop_gaps"]
    elif has_demand and has_receipt:
        intent = "cross_file_risk"
        sources = ["item_master", "demand_history", "receipt_history", "analysis_results"]
        operations = ["aggregate_demand_by_sku", "aggregate_receipts_by_sku", "combine_at_sku_grain", "filter_and_rank"]
    elif suppliers and has_demand:
        intent = "supplier_demand"
        sources = ["item_master", "demand_history"]
        operations = ["resolve_supplier_skus", "aggregate_demand_by_sku", "filter_and_rank"]
    elif suppliers and has_receipt:
        intent = "supplier_receipts"
        sources = ["item_master", "receipt_history"]
        operations = ["resolve_supplier_skus", "aggregate_receipts_by_sku", "filter_and_rank"]
    elif suppliers and re.search(r"what skus|which skus|products|items", q):
        intent = "supplier_skus"
        sources = ["item_master"]
        operations = ["resolve_supplier_skus"]
    elif suppliers or asks_about_suppliers:
        intent = "supplier_overview"
        sources = ["item_master", "demand_history", "receipt_history", "analysis_results"]
        operations = ["build_supplier_profiles", "rank_suppliers"]
    elif has_receipt:
        intent = "sku_receipts" if skus else "receipt_ranking"
        sources = ["receipt_history"] + (["item_master"] if not skus else [])
        operations = ["aggregate_receipts_by_sku", "retrieve_receipt_metrics"]
    elif has_demand:
        intent = "sku_demand" if skus else "demand_ranking"
        sources = ["demand_history"] + (["item_master"] if not skus else [])
        operations = ["select_time_window", "aggregate_demand_by_sku", "filter_and_rank"]
    elif has_risk:
        intent = "stockout_risk"
        sources = ["item_master", "demand_history", "receipt_history", "analysis_results"]
        operations = ["aggregate_each_fact_by_sku", "calculate_relative_stockout_risk", "rank"]
    elif skus:
        intent = "sku_profile"
        sources = ["item_master", "demand_history", "receipt_history", "analysis_results"]
        operations = ["build_sku_profile"]
    else:
        intent = "general_inventory"
        sources = ["item_master", "demand_history", "receipt_history", "analysis_results"]
        operations = ["aggregate_each_fact_by_sku", "rank_portfolio_risk"]

    filters: dict[str, Any] = {}
    if suppliers:
        filters["supplier"] = suppliers
    if skus:
        filters["sku"] = skus
    if re.search(r"\b(?:class\s+)?a(?:-class)?\b", q):
        filters["item_class"] = "A"
    return QuestionPlan(intent, skus, suppliers, window, sources, operations, filters, ambiguous_entities=ambiguous)


def detect_intent(question: str, prior_intent: str | None = None) -> str:
    """Compatibility intent detector; full planning is handled by plan_question."""
    q = question.casefold().strip()
    if re.search(r"otif|fill rate|promised[- ]date|historical stockout", q):
        return "unsupported"
    if "forecast" in q or "predict" in q:
        return "forecast"
    if "safety stock" in q or re.search(r"\bss\b", q):
        return "safety_stock"
    if "reorder point" in q or re.search(r"\brop\b", q):
        return "reorder_point"
    if "stockout" in q or "risk" in q:
        return "stockout_risk"
    if "supplier" in q or "vendor" in q:
        return "supplier_performance"
    if "receipt" in q or "lead time" in q or "lead-time" in q:
        return "receipt_performance"
    if "demand" in q or "volatile" in q or "trend" in q:
        return "demand"
    if q in {"why", "why?", "explain", "tell me more"} and prior_intent:
        return prior_intent
    return "general_inventory"


def _records(frame: pd.DataFrame, limit: int = MAX_CONTEXT_ROWS) -> list[dict[str, Any]]:
    selected = frame.head(limit)
    clean = selected.astype(object).where(pd.notna(selected), None)
    return json.loads(clean.to_json(orient="records"))


def _columns(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    return frame[[name for name in names if name in frame.columns]]


def _filter_view(view: pd.DataFrame, plan: QuestionPlan) -> pd.DataFrame:
    result = view.copy()
    if plan.skus:
        wanted = {value.casefold() for value in plan.skus}
        result = result.loc[result["sku"].astype(str).str.casefold().isin(wanted)]
    if plan.suppliers:
        wanted = {value.casefold() for value in plan.suppliers}
        result = result.loc[result["supplier"].astype(str).str.casefold().isin(wanted)]
    if plan.filters.get("item_class"):
        result = result.loc[
            result["item_class"].astype(str).str.upper() == plan.filters["item_class"]
        ]
    return result


def _cross_file_ranking(view: pd.DataFrame, question: str, limit: int) -> tuple[pd.DataFrame, dict]:
    q = question.casefold()
    result = view.copy()
    demand_high = result["demand_cv"].quantile(0.75)
    demand_low = result["demand_cv"].quantile(0.25)
    lead_high = result["lead_time_cv"].quantile(0.75)
    lead_low = result["lead_time_cv"].quantile(0.25)
    definitions = {
        "high_demand_variability": f"demand CV at or above portfolio 75th percentile ({demand_high:.3f})",
        "stable_demand": f"demand CV at or below portfolio 25th percentile ({demand_low:.3f})",
        "high_lead_time_variability": f"lead-time CV at or above portfolio 75th percentile ({lead_high:.3f})",
        "stable_lead_time": f"lead-time CV at or below portfolio 25th percentile ({lead_low:.3f})",
        "increasing_demand": "recent 8-period average is more than 10% above the prior comparable period",
        "low_current_safety_stock": "current Safety Stock at or below the selected portfolio's 25th percentile",
    }
    if re.search(r"rising|increas|growing", q):
        result = result.loc[result["demand_trend"] == "increasing"]
    if re.search(r"highly variable lead|unstable lead|inconsistent receipt|supply.*risk", q):
        result = result.loc[result["lead_time_cv"] >= lead_high]
    if re.search(r"stable (?:historical )?demand", q):
        result = result.loc[result["demand_cv"] <= demand_low]
    elif re.search(r"volatile demand|demand.*volatile", q):
        result = result.loc[result["demand_cv"] >= demand_high]
    if re.search(r"stable lead|consistent receipt", q):
        result = result.loc[result["lead_time_cv"] <= lead_low]
    if re.search(r"longer-than-average|longer than average", q):
        result = result.loc[result["lead_time_mean"] > view["lead_time_mean"].mean()]
    if re.search(r"low current safety stock|low safety stock|low current ss", q):
        low_ss = view["current_safety_stock"].quantile(0.25)
        result = result.loc[result["current_safety_stock"] <= low_ss]
    if re.search(r"both|demand side.*supply|risk from.*demand.*supply", q) and result.equals(view):
        result = result.loc[(result["demand_cv"] >= demand_high) & (result["lead_time_cv"] >= lead_high)]
    sort_column = "safety_stock_gap" if "safety stock" in q or re.search(r"\bss\b", q) else "stockout_risk_score"
    columns = [
        "sku", "description", "supplier", "item_class", "demand_recent_avg",
        "demand_previous_avg", "demand_change_pct", "demand_trend", "demand_cv",
        "lead_time_mean", "lead_time_std", "lead_time_cv", "current_safety_stock",
        "recommended_safety_stock", "safety_stock_gap", "stockout_risk_score",
    ]
    return _columns(result.sort_values(sort_column, ascending=False).head(limit), columns), definitions


def _provenance_lines(sources: Sequence[str]) -> str:
    descriptions = {
        "item_master": "SKU attributes, ABC class, current parameters, and current primary supplier",
        "demand_history": "uploaded period-level demand aggregated by SKU",
        "receipt_history": "uploaded PO receipt lead times aggregated by SKU",
        "analysis_results": "calculated Safety Stock/ROP recommendations, gaps, and flags",
    }
    return "\n\nBased on:\n" + "\n".join(
        f"- {source}: {descriptions[source]}" for source in sources if source in descriptions
    )


def _build_evidence(
    question: str,
    plan: QuestionPlan,
    model: InventoryDataModel,
) -> tuple[dict[str, Any], pd.DataFrame | None, str | None, list[str], dict[str, Any]]:
    q = question.casefold()
    limit = _limit_from_question(question)
    facts: dict[str, Any] = {}
    definitions: dict[str, Any] = {}
    limitations = list(plan.limitations)
    table: pd.DataFrame | None = None
    direct_answer: str | None = None

    if plan.ambiguous_entities:
        direct_answer = "The entity reference is ambiguous. Please specify one of: " + "; ".join(plan.ambiguous_entities)
        limitations.append("No data query was executed because entity resolution was ambiguous.")
    elif plan.intent == "unknown_entity":
        direct_answer = plan.limitations[0] + ". Please check the identifier and try again."
    elif plan.intent == "unsupported":
        if "stockout" in q or "stock out" in q:
            direct_answer = (
                "Historical stockout events are not available in the uploaded datasets. "
                "I can evaluate relative Stockout Risk from demand, lead-time, and planning gaps."
            )
        else:
            direct_answer = (
                "The uploaded data cannot establish OTIF, fill rate, supplier quality, or "
                "promised-date adherence."
            )
        limitations.append(direct_answer)
    elif plan.intent == "supplier_lookup":
        if not plan.skus:
            direct_answer = "Please specify a known SKU."
        else:
            table = _columns(
                model.item_master.loc[model.item_master["sku"].isin(plan.skus)],
                ["sku", "description", "supplier", "item_class"],
            )
            if table.empty:
                direct_answer = f"SKU {plan.skus[0]} was not found in item_master."
            else:
                facts["item_master_lookup"] = _records(table)
    elif plan.intent == "sku_demand":
        if not plan.skus:
            direct_answer = "Please specify a known SKU for demand analysis."
        else:
            metrics, table = model.demand_window(plan.skus[0], plan.time_window or 8)
            facts["demand_metrics"] = metrics
            definitions["recent_comparison"] = (
                "most recent requested periods versus the immediately preceding comparable periods"
            )
            if metrics.get("limitation"):
                limitations.append(metrics["limitation"])
            if metrics.get("ordering_limitation"):
                limitations.append(metrics["ordering_limitation"])
    elif plan.intent == "forecast":
        if not plan.skus:
            direct_answer = "Please specify a known SKU for the demand forecast."
        else:
            horizon_match = re.search(r"\bnext\s+(\d{1,2})\s+(?:weeks?|periods?)\b", q)
            horizon = int(horizon_match.group(1)) if horizon_match else 4
            forecast = forecast_demand(model.demand_history, plan.skus[0], horizon)
            table = forecast.pop("forecast")
            facts["forecast"] = forecast
            facts["recent_demand"] = model.demand_window(plan.skus[0], min(8, horizon))[0]
            definitions["forecast_method"] = forecast.get("method")
            if forecast.get("limitation"):
                limitations.append(forecast["limitation"])
    elif plan.intent == "sku_receipts":
        if not plan.skus:
            direct_answer = "Please specify a known SKU for receipt analysis."
        else:
            metrics = model.receipt_metrics(plan.skus[0])
            facts["receipt_metrics"] = metrics
            table = pd.DataFrame([metrics])
            if re.search(r"recent|trend|slower|faster|increas|decreas", q):
                limitations.append(metrics.get("chronology_limitation", "Receipt chronology is unavailable."))
    elif plan.intent in {"sku_safety_stock", "sku_rop", "sku_profile"}:
        if not plan.skus:
            direct_answer = "Please specify a known SKU."
        else:
            profile = model.sku_profile(plan.skus[0], plan.time_window or 8)
            if profile is None:
                direct_answer = f"SKU {plan.skus[0]} was not found."
            else:
                facts["sku_profile"] = profile
                row = _filter_view(model.sku_view, plan)
                table = _columns(row, [
                    "sku", "description", "supplier", "item_class", "demand_recent_avg",
                    "demand_previous_avg", "demand_change_pct", "demand_cv", "lead_time_mean",
                    "lead_time_std", "lead_time_cv", "current_safety_stock",
                    "recommended_safety_stock", "safety_stock_gap", "current_rop",
                    "recommended_rop", "rop_gap", "analysis_flags", "stockout_risk_score",
                ])
                definitions["safety_stock_gap"] = "recommended Safety Stock minus current Safety Stock"
                definitions["rop_gap"] = "recommended ROP minus current ROP"
    elif plan.intent == "supplier_skus":
        if not plan.suppliers:
            direct_answer = "Please specify a known supplier."
        else:
            table = _columns(
                model.item_master.loc[model.item_master["supplier"].isin(plan.suppliers)],
                ["sku", "description", "supplier", "item_class"],
            ).head(limit)
            facts["supplier_skus"] = _records(table)
            definitions["supplier_mapping"] = "current primary supplier from item_master"
    elif plan.intent in {"supplier_demand", "supplier_receipts", "supplier_inventory"}:
        if not plan.suppliers:
            direct_answer = "Please specify a known supplier."
        else:
            rows = _filter_view(model.sku_view, plan)
            if plan.intent == "supplier_demand":
                if re.search(r"increas|growing|rising", q):
                    rows = rows.loc[rows["demand_trend"] == "increasing"]
                table = _columns(rows.sort_values("demand_cv", ascending=False), [
                    "sku", "description", "supplier", "item_class", "demand_mean",
                    "demand_recent_avg", "demand_previous_avg", "demand_change_pct",
                    "demand_trend", "demand_std", "demand_cv",
                ]).head(limit)
                definitions["demand_cv"] = "demand standard deviation divided by mean demand"
            elif plan.intent == "supplier_receipts":
                table = _columns(rows.sort_values("lead_time_cv", ascending=False), [
                    "sku", "description", "supplier", "item_class", "receipt_count",
                    "lead_time_mean", "lead_time_median", "lead_time_std", "lead_time_cv",
                    "lead_time_min", "lead_time_max", "lead_time_outlier_count",
                ]).head(limit)
                definitions["lead_time_cv"] = "lead-time standard deviation divided by mean lead time"
                if re.search(r"recent|trend|slower|faster|increas|decreas", q):
                    limitations.append(
                        "receipt_history has no order/receipt date; recent supplier slowdown cannot be established."
                    )
            else:
                table = _columns(rows.sort_values("safety_stock_gap", ascending=False), [
                    "sku", "description", "supplier", "item_class", "demand_cv",
                    "demand_trend", "lead_time_cv", "current_safety_stock",
                    "recommended_safety_stock", "safety_stock_gap", "current_rop",
                    "recommended_rop", "rop_gap", "stockout_risk_score",
                ]).head(limit)
                definitions["safety_stock_gap"] = "recommended Safety Stock minus current Safety Stock"
            facts["supplier_sku_results"] = _records(table)
    elif plan.intent in {"supplier_comparison", "supplier_overview"}:
        summary = model.supplier_summary()
        if plan.suppliers:
            wanted = {value.casefold() for value in plan.suppliers}
            summary = summary.loc[summary["supplier"].str.casefold().isin(wanted)]
        if ("under-protected" in q or "under protected" in q) and (
            "concentration" in q or "share" in q or "percent" in q
        ):
            summary = summary.sort_values("under_protected_a_share", ascending=False)
        elif "a-class" in q or "class a" in q:
            summary = summary.assign(
                a_class_skus=summary["supplier"].map(
                    lambda supplier: int((
                        (model.sku_view["supplier"] == supplier)
                        & (model.sku_view["item_class"].astype(str).str.upper() == "A")
                    ).sum())
                )
            ).sort_values("a_class_skus", ascending=False)
        elif "under-protected" in q or "under protected" in q:
            summary = summary.sort_values("under_protected_a_skus", ascending=False)
        elif "safety stock" in q or re.search(r"\bss\b", q):
            summary = summary.sort_values("total_positive_safety_stock_gap_units", ascending=False)
        elif "consistent" in q or "variab" in q:
            summary = summary.sort_values("receipt_level_lead_time_cv", ascending=True)
        elif "risk" in q:
            summary = summary.sort_values("average_stockout_risk_score_across_skus", ascending=False)
        table = summary.head(limit)
        facts["supplier_profiles"] = _records(table)
        definitions.update({
            "lead_time_average": "weighted across individual receipt records",
            "risk_average": "simple average across supplier SKUs",
            "supplier_mapping": "current primary supplier from item_master",
            "positive_safety_stock_gap": "sum of positive recommended-minus-current Safety Stock gaps by supplier",
            "under_protected_a_share": "under-protected A-class SKUs divided by all A-class SKUs for each supplier",
        })
    elif plan.intent == "sku_comparison":
        table = _columns(_filter_view(model.sku_view, plan), [
            "sku", "description", "supplier", "item_class", "demand_mean", "demand_cv",
            "demand_change_pct", "lead_time_mean", "lead_time_cv", "current_safety_stock",
            "recommended_safety_stock", "safety_stock_gap", "current_rop", "recommended_rop",
            "rop_gap", "stockout_risk_score",
        ])
        facts["sku_comparison"] = _records(table)
    elif plan.intent == "demand_ranking":
        rows = _filter_view(model.sku_view, plan)
        if re.search(r"increas|growing|rising", q):
            rows = rows.loc[rows["demand_trend"] == "increasing"].sort_values(
                "demand_change_pct", ascending=False
            )
        else:
            rows = rows.sort_values("demand_cv", ascending=False)
        table = _columns(rows, [
            "sku", "description", "supplier", "item_class", "demand_history_points",
            "demand_mean", "demand_std", "demand_cv", "demand_recent_avg",
            "demand_previous_avg", "demand_change_pct", "demand_trend",
        ]).head(limit)
        facts["demand_ranking"] = _records(table)
    elif plan.intent == "receipt_ranking":
        rows = _filter_view(model.sku_view, plan)
        if re.search(r"recent|trend|slower|faster|increas|decreas", q):
            limitations.append(
                "receipt_history has no order/receipt date; recent lead-time direction cannot be established."
            )
        sort_col = "lead_time_cv" if re.search(r"variab|inconsistent|consistent", q) else "lead_time_mean"
        table = _columns(rows.sort_values(sort_col, ascending=False), [
            "sku", "description", "supplier", "item_class", "receipt_count",
            "lead_time_mean", "lead_time_median", "lead_time_std", "lead_time_cv",
            "lead_time_min", "lead_time_max", "lead_time_outlier_count",
        ]).head(limit)
        facts["receipt_ranking"] = _records(table)
    elif plan.intent == "cross_file_risk":
        rows = _filter_view(model.sku_view, plan)
        table, cross_definitions = _cross_file_ranking(rows, question, limit)
        definitions.update(cross_definitions)
        facts["cross_file_result"] = _records(table)
    elif plan.intent == "inventory_ranking":
        rows = _filter_view(model.sku_view, plan)
        sort_col = "rop_gap" if "rop" in q or "reorder" in q else "safety_stock_gap"
        ascending = bool(re.search(r"decreas|excess|reduce|lower", q))
        table = _columns(rows.sort_values(sort_col, ascending=ascending), [
            "sku", "description", "supplier", "item_class", "demand_cv", "lead_time_cv",
            "current_safety_stock", "recommended_safety_stock", "safety_stock_gap",
            "current_rop", "recommended_rop", "rop_gap", "stockout_risk_score",
        ]).head(limit)
        facts["inventory_gap_ranking"] = _records(table)
    else:
        rows = _filter_view(model.sku_view, plan).sort_values("stockout_risk_score", ascending=False)
        table = _columns(rows, [
            "sku", "description", "supplier", "item_class", "demand_cv", "demand_trend",
            "lead_time_cv", "safety_stock_gap", "rop_gap", "stockout_risk_score",
            "stockout_risk_level",
        ]).head(limit)
        facts["portfolio_risk"] = _records(table)
        definitions["stockout_risk"] = (
            "relative planning score combining Python-calculated demand, lead-time, gap, trend, and ABC factors"
        )

    if table is not None and table.empty and direct_answer is None:
        limitations.append("No rows matched the requested filters.")
    debug = {
        "intent": plan.intent,
        "entities": {"skus": plan.skus, "suppliers": plan.suppliers},
        "sources_used": plan.required_sources,
        "operations": plan.operations,
        "rows_sent_to_llm": 0 if table is None else min(len(table), MAX_CONTEXT_ROWS),
    }
    return facts, table, direct_answer, limitations, {"definitions": definitions, "debug": debug}


def build_analytical_context(
    question: str,
    item_master: pd.DataFrame,
    demand_history: pd.DataFrame,
    receipt_history: pd.DataFrame,
    analysis_results: Sequence[Mapping[str, Any]] | pd.DataFrame,
    prior_metadata: Mapping[str, Any] | None = None,
    data_model: InventoryDataModel | None = None,
) -> tuple[str, pd.DataFrame | None, dict[str, Any], str | None]:
    """Plan, execute, and serialize one question-specific evidence package."""
    model = data_model or InventoryDataModel.build(
        item_master, demand_history, receipt_history, analysis_results
    )
    plan = plan_question(question, model.item_master, prior_metadata)
    facts, table, direct_answer, limitations, extra = _build_evidence(question, plan, model)
    focus_sku = str(table.iloc[0]["sku"]) if table is not None and not table.empty and "sku" in table else (
        plan.skus[0] if plan.skus else None
    )
    focus_supplier = (
        str(table.iloc[0]["supplier"])
        if table is not None and not table.empty and "supplier" in table
        else plan.suppliers[0] if plan.suppliers else None
    )
    evidence = {
        "question_plan": plan.to_dict(),
        "data_sources_used": plan.required_sources,
        "filters": plan.filters,
        "calculated_result": facts,
        "definitions": extra["definitions"],
        "limitations": list(dict.fromkeys(limitations)),
        "debug": extra["debug"],
    }
    metadata = {
        "intent": plan.intent,
        "skus": plan.skus,
        "suppliers": plan.suppliers,
        "focus_sku": focus_sku,
        "focus_supplier": focus_supplier,
        "sources_used": plan.required_sources,
        "operations": plan.operations,
        "evidence": evidence,
    }
    context = (
        "QUESTION\n" + question
        + "\n\nQUESTION PLAN\n" + json.dumps(plan.to_dict(), indent=2, default=str)
        + "\n\nDATA SOURCES USED\n" + "\n".join(f"- {source}" for source in plan.required_sources)
        + "\n\nFILTERS\n" + json.dumps(plan.filters, indent=2, default=str)
        + "\n\nPYTHON-GENERATED EVIDENCE\n" + json.dumps(facts, indent=2, default=str)
        + "\n\nDEFINITIONS\n" + json.dumps(extra["definitions"], indent=2, default=str)
        + "\n\nLIMITATIONS\n" + json.dumps(evidence["limitations"], indent=2, default=str)
    )
    if direct_answer:
        direct_answer += _provenance_lines(plan.required_sources)
    return context, table, metadata, direct_answer


def _safe_api_error(exc: Exception) -> InventoryAssistantError:
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    if "auth" in name or "401" in message or "api key" in message:
        detail = "NVIDIA authentication failed. Check NVIDIA_API_KEY."
    elif "rate" in name or "429" in message or "quota" in message:
        detail = "The NVIDIA API rate limit or quota was reached. Try again later."
    elif "timeout" in name or "timed out" in message:
        detail = "The NVIDIA API timed out. Try the question again."
    elif "model" in message and ("invalid" in message or "not found" in message):
        detail = "The configured NVIDIA_MODEL is unavailable or invalid."
    else:
        detail = "The NVIDIA assistant request failed. Your inventory analysis is still available."
    return InventoryAssistantError(detail)


def ask_inventory_assistant(
    question: str,
    item_master: pd.DataFrame,
    demand_history: pd.DataFrame,
    receipt_history: pd.DataFrame,
    analysis_results: Sequence[Mapping[str, Any]] | pd.DataFrame,
    chat_history: Sequence[Mapping[str, str]] | None = None,
    prior_metadata: Mapping[str, Any] | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    client: Any | None = None,
    data_model: InventoryDataModel | None = None,
) -> AssistantAnswer:
    if not question or not question.strip():
        raise ValueError("Enter a question about the completed inventory analysis.")
    try:
        context, table, metadata, direct_answer = build_analytical_context(
            question, item_master, demand_history, receipt_history, analysis_results,
            prior_metadata, data_model,
        )
    except InventoryDataError as exc:
        raise InventoryAssistantError(str(exc)) from exc
    evidence = metadata.get("evidence", {})
    if direct_answer:
        return AssistantAnswer(direct_answer, table, metadata, evidence)

    config = get_nvidia_config(api_key, model, base_url)
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise InventoryAssistantError(
                "The OpenAI-compatible client is not installed. Install project requirements."
            ) from exc
        client = OpenAI(
            base_url=config["base_url"], api_key=config["api_key"], timeout=30.0, max_retries=1
        )

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in list(chat_history or [])[-MAX_HISTORY_MESSAGES:]:
        role, content = message.get("role"), message.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": str(content)[:2000]})
    messages.append({
        "role": "user",
        "content": context + "\n\nExplain only this evidence. The deterministic table is displayed separately.",
    })
    try:
        response = client.chat.completions.create(
            model=config["model"], messages=messages, temperature=0.2, max_tokens=700
        )
        response_text = response.choices[0].message.content
        if not response_text or not str(response_text).strip():
            raise InventoryAssistantError("NVIDIA returned an empty response. Try again.")
    except InventoryAssistantError:
        raise
    except Exception as exc:
        raise _safe_api_error(exc) from exc
    text = str(response_text).strip() + _provenance_lines(metadata.get("sources_used", []))
    return AssistantAnswer(text, table, metadata, evidence)

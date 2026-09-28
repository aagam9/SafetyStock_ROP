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
from inventory_question_parser import (
    LanguageInterpretation,
    extract_unresolved_sku_candidate,
    interpret_inventory_question,
    validate_interpretation_payload,
)


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
    query_type: str = "lookup"
    entity_type: str = "sku"
    metric: str | None = None
    metrics: list[str] = field(default_factory=list)
    direction: str | None = None
    limit: int = 10
    conditions: list[str] = field(default_factory=list)
    confidence: str = "high"
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
    skus: list[str] = []
    for value in item_master["sku"].dropna().unique():
        sku = str(value)
        normalized_sku = _normalize_entity(sku)
        # Very short codes such as A/B are valid only with explicit SKU syntax;
        # otherwise ordinary prose and "A-class" would create false matches.
        if len(normalized_sku) < 3:
            explicitly_named = bool(re.search(
                rf"\bsku(?:\s+(?:code|id|number))?\s*[:#]?\s+{re.escape(sku.casefold())}\b",
                q,
            ))
            if explicitly_named or q.strip(" ?.!") == sku.casefold():
                skus.append(sku)
        elif _boundary_match(q, sku):
            skus.append(sku)
    suppliers = [
        str(value) for value in item_master["supplier"].dropna().unique()
        if _boundary_match(q, str(value))
    ]
    descriptions = item_master[["sku", "description"]].dropna()
    for _, row in descriptions.iterrows():
        description = str(row["description"])
        if len(description) >= 4 and _boundary_match(q, description):
            skus.append(str(row["sku"]))

    # Normalized matching is only applied to known, sufficiently distinctive
    # identifiers. Arbitrary words are never promoted to SKU values.
    if not skus:
        normalized_matches = [
            str(value) for value in item_master["sku"].dropna().unique()
            if len(_normalize_entity(str(value))) >= 4
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

    # If the same unqualified phrase can name both an SKU/description and a
    # supplier alias, do not guess. Explicit "SKU ..." or "Supplier ..."
    # phrasing disambiguates it.
    if skus and suppliers:
        sku_labels: set[str] = set()
        for sku in skus:
            sku_labels.add(_normalize_entity(sku))
            descriptions = item_master.loc[item_master["sku"].astype(str) == sku, "description"]
            sku_labels.update(_normalize_entity(value) for value in descriptions.dropna())
        supplier_labels: set[str] = set()
        for supplier in suppliers:
            supplier_labels.add(_normalize_entity(supplier))
            alias = re.sub(r"^(supplier|vendor)\s+", "", supplier, flags=re.I).strip()
            supplier_labels.add(_normalize_entity(alias))
        collisions = {value for value in sku_labels & supplier_labels if len(value) >= 3}
        explicit_sku_type = bool(re.search(r"\bsku\s+[a-z0-9]", q))
        explicit_supplier_type = bool(re.search(r"\b(supplier|vendor)\s+[a-z0-9]", q))
        if collisions:
            if explicit_supplier_type and not explicit_sku_type:
                skus = []
            elif explicit_sku_type and not explicit_supplier_type:
                suppliers = []
            elif not explicit_sku_type and not explicit_supplier_type:
                ambiguous.append(
                    "entity may refer to an SKU/description or supplier: "
                    + ", ".join(sorted(collisions))
                )
                skus = []
                suppliers = []

    prior = dict(prior_metadata or {})
    references_prior = bool(
        re.fullmatch(r"\s*(why\??|explain|tell me more)\s*", q)
        or re.search(r"\b(it|its|that one|this sku|that sku|this item|that item)\b", q)
    )
    resolved_follow_up_sku = False
    if references_prior:
        focus_sku = prior.get("focus_sku") or next(iter(prior.get("skus", [])), None)
        if focus_sku and str(focus_sku) not in skus:
            skus.append(str(focus_sku))
        resolved_follow_up_sku = bool(skus)
    # A ranked result may carry both SKU and supplier metadata. Pronouns such as
    # "it" and a bare "Why?" refer to the focused SKU first; only fall back to
    # supplier context when there is no focused SKU.
    if references_prior and not suppliers and not resolved_follow_up_sku and not skus:
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
    interpretation_override: LanguageInterpretation | None = None,
) -> QuestionPlan:
    """Create a validated plan selecting only approved sources and operations."""
    q = question.casefold()
    skus, suppliers, ambiguous = _resolve_entities(question, item_master, prior_metadata)
    window = _time_window(question)
    interpretation = interpretation_override or interpret_inventory_question(
        question, sku_count=len(skus), supplier_count=len(suppliers), maximum_rows=MAX_CONTEXT_ROWS
    )

    def make_plan(
        intent: str,
        sources: list[str],
        operations: list[str],
        *,
        limitations: list[str] | None = None,
    ) -> QuestionPlan:
        filters: dict[str, Any] = {}
        if suppliers:
            filters["supplier"] = suppliers
        if skus:
            filters["sku"] = skus
        if re.search(r"\b(?:class\s+)?a(?:-class)?\b", q):
            filters["item_class"] = "A"
        return QuestionPlan(
            intent=intent,
            skus=skus,
            suppliers=suppliers,
            time_window=window,
            required_sources=list(dict.fromkeys(sources)),
            operations=operations,
            query_type=interpretation.query_type,
            entity_type=interpretation.entity_type,
            metric=interpretation.metric,
            metrics=list(interpretation.metrics),
            direction=interpretation.direction,
            limit=interpretation.limit,
            conditions=list(interpretation.conditions),
            confidence=interpretation.confidence,
            filters=filters,
            limitations=list(limitations or []),
            ambiguous_entities=ambiguous,
        )

    if ambiguous:
        return make_plan("clarification", [], ["resolve_ambiguous_entity"])

    unresolved_sku = extract_unresolved_sku_candidate(question)
    if unresolved_sku and not skus:
        return make_plan(
            "unknown_entity", ["item_master"], ["resolve_sku"],
            limitations=[f"Unknown SKU: {unresolved_sku}"],
        )
    unsupported = re.search(r"\b(otif|fill rate|quality|promised[- ]date|on[- ]time delivery)\b", q)
    historical_stockout = (
        ("stockout" in q or "stock out" in q)
        and re.search(r"\b(how many|history|historical|last year|occurred|events?)\b", q)
        and "risk" not in q
    )
    if unsupported or historical_stockout:
        return make_plan("unsupported", [], ["report_limitation"])

    metric = interpretation.metric
    metrics = set(interpretation.metrics)
    asks_for_supplier = bool(re.search(
        r"\b(what|which)\s+supplier\b|\bwho\s+supplies\b|\bwhat\s+vendor\b|\bsupplier\s+is\s+it\s+from\b",
        q,
    ))
    if skus and asks_for_supplier and len(skus) == 1 and interpretation.query_type != "comparison":
        return make_plan("supplier_lookup", ["item_master"], ["lookup_item_master"])

    if interpretation.query_type == "forecast":
        return make_plan("forecast", ["demand_history"], ["resolve_sku", "fit_deterministic_forecast"])
    if interpretation.query_type == "comparison":
        if len(suppliers) >= 2 and not skus:
            return make_plan(
                "supplier_comparison",
                ["item_master", "demand_history", "receipt_history", "analysis_results"],
                ["resolve_supplier_skus", "aggregate_each_fact_by_sku", "compare_supplier_profiles"],
            )
        if len(skus) >= 2:
            return make_plan(
                "sku_comparison",
                ["item_master", "demand_history", "receipt_history", "analysis_results"],
                ["build_sku_profiles", "compare_profiles"],
            )
        return make_plan(
            "clarification", [], ["request_comparison_entities"],
            limitations=["Please specify two known SKUs or two known suppliers to compare."],
        )

    domains = set()
    if metrics & {"demand_variability", "demand_level", "demand_trend"}:
        domains.add("demand")
    if metrics & {"lead_time_variability", "lead_time_mean"}:
        domains.add("receipt")
    if metrics & {"safety_stock_gap", "rop_gap", "stockout_risk"}:
        domains.add("inventory")
    cross_file = len(domains) >= 2

    if cross_file:
        sources = ["item_master"]
        if "demand" in domains:
            sources.append("demand_history")
        if "receipt" in domains:
            sources.append("receipt_history")
        if "inventory" in domains:
            sources.append("analysis_results")
        return make_plan(
            "cross_file_risk", sources,
            ["aggregate_required_facts_by_sku", "combine_at_sku_grain", "apply_conditions", "rank"],
        )

    if metric in {"safety_stock_gap", "rop_gap"}:
        if skus:
            intent = "sku_safety_stock" if metric == "safety_stock_gap" else "sku_rop"
            sources = ["item_master", "demand_history", "receipt_history", "analysis_results"]
            return make_plan(intent, sources, ["build_sku_profile", "explain_inventory_drivers"])
        if suppliers and interpretation.entity_type == "sku":
            return make_plan(
                "supplier_inventory", ["item_master", "analysis_results"],
                ["resolve_supplier_skus", "rank_inventory_gaps"],
            )
        if interpretation.entity_type == "supplier":
            return make_plan(
                "supplier_overview", ["item_master", "analysis_results"],
                ["build_supplier_profiles", "rank_suppliers"],
            )
        return make_plan(
            "inventory_ranking", ["item_master", "analysis_results"],
            ["rank_safety_stock_gaps" if metric == "safety_stock_gap" else "rank_rop_gaps"],
        )

    if metric in {"lead_time_variability", "lead_time_mean"}:
        if interpretation.entity_type == "supplier" and not skus:
            return make_plan(
                "supplier_overview", ["item_master", "receipt_history"],
                ["map_receipts_to_current_supplier", "rank_supplier_lead_time_metric"],
            )
        if skus:
            return make_plan("sku_receipts", ["receipt_history"], ["retrieve_receipt_metrics"])
        if suppliers:
            return make_plan(
                "supplier_receipts", ["item_master", "receipt_history"],
                ["resolve_supplier_skus", "aggregate_receipts_by_sku", "rank"],
            )
        return make_plan(
            "receipt_ranking", ["receipt_history"],
            ["aggregate_receipts_by_sku", "rank_lead_time_metric"],
        )

    if metric in {"demand_variability", "demand_level", "demand_trend"}:
        if skus:
            return make_plan("sku_demand", ["demand_history"], ["select_time_window", "calculate_demand_metrics"])
        if suppliers:
            return make_plan(
                "supplier_demand", ["item_master", "demand_history"],
                ["resolve_supplier_skus", "aggregate_demand_by_sku", "rank"],
            )
        return make_plan(
            "demand_ranking", ["demand_history"],
            ["aggregate_demand_by_sku", "rank_demand_metric"],
        )

    if metric == "stockout_risk":
        return make_plan(
            "stockout_risk", ["item_master", "demand_history", "receipt_history", "analysis_results"],
            ["aggregate_each_fact_by_sku", "calculate_relative_stockout_risk", "rank"],
        )

    if suppliers and interpretation.entity_type == "sku":
        return make_plan("supplier_skus", ["item_master"], ["resolve_supplier_skus"])
    if suppliers or interpretation.entity_type == "supplier":
        return make_plan(
            "supplier_overview", ["item_master", "demand_history", "receipt_history", "analysis_results"],
            ["build_supplier_profiles"],
        )
    if skus:
        return make_plan(
            "sku_profile", ["item_master", "demand_history", "receipt_history", "analysis_results"],
            ["build_sku_profile"],
        )
    return make_plan(
        "clarification", [], ["request_metric_or_entity"],
        limitations=[interpretation.clarification or "Please specify an inventory metric or entity."],
    )


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


def _apply_single_domain_conditions(frame: pd.DataFrame, plan: QuestionPlan) -> pd.DataFrame:
    """Apply deterministic percentile/trend filters within the selected scope."""
    result = frame.copy()
    conditions = set(plan.conditions)
    if "high_demand_variability" in conditions:
        result = result.loc[result["demand_cv"] >= result["demand_cv"].quantile(0.75)]
    if "stable_demand" in conditions:
        result = result.loc[result["demand_cv"] <= result["demand_cv"].quantile(0.25)]
    if "increasing_demand" in conditions:
        result = result.loc[result["demand_trend"] == "increasing"]
    if "high_lead_time_variability" in conditions:
        result = result.loc[result["lead_time_cv"] >= result["lead_time_cv"].quantile(0.75)]
    if "stable_lead_time" in conditions:
        result = result.loc[result["lead_time_cv"] <= result["lead_time_cv"].quantile(0.25)]
    if "long_lead_time" in conditions:
        result = result.loc[result["lead_time_mean"] > result["lead_time_mean"].mean()]
    return result


def _cross_file_ranking(
    view: pd.DataFrame, plan: QuestionPlan, limit: int
) -> tuple[pd.DataFrame, dict]:
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
    conditions = set(plan.conditions)
    if "increasing_demand" in conditions:
        result = result.loc[result["demand_trend"] == "increasing"]
    if "high_lead_time_variability" in conditions:
        result = result.loc[result["lead_time_cv"] >= lead_high]
    if "stable_demand" in conditions:
        result = result.loc[result["demand_cv"] <= demand_low]
    elif "high_demand_variability" in conditions:
        result = result.loc[result["demand_cv"] >= demand_high]
    if "stable_lead_time" in conditions:
        result = result.loc[result["lead_time_cv"] <= lead_low]
    if "long_lead_time" in conditions:
        result = result.loc[result["lead_time_mean"] > view["lead_time_mean"].mean()]
    if "positive_safety_stock_gap" in conditions:
        result = result.loc[result["safety_stock_gap"] > 0]
    if "positive_rop_gap" in conditions:
        result = result.loc[result["rop_gap"] > 0]
    if "low_current_safety_stock" in conditions:
        low_ss = view["current_safety_stock"].quantile(0.25)
        result = result.loc[result["current_safety_stock"] <= low_ss]
    if {"demand_variability", "lead_time_variability"}.issubset(set(plan.metrics)) and not (
        {"high_demand_variability", "high_lead_time_variability"} & conditions
    ):
        result = result.loc[(result["demand_cv"] >= demand_high) & (result["lead_time_cv"] >= lead_high)]
    sort_columns = {
        "demand_variability": "demand_cv",
        "demand_level": "demand_mean",
        "demand_trend": "demand_change_pct",
        "lead_time_variability": "lead_time_std",
        "lead_time_mean": "lead_time_mean",
        "safety_stock_gap": "safety_stock_gap",
        "rop_gap": "rop_gap",
        "stockout_risk": "stockout_risk_score",
    }
    sort_column = sort_columns.get(plan.metric, "stockout_risk_score")
    ascending = plan.direction == "ascending"
    columns = [
        "sku", "description", "supplier", "item_class", "demand_recent_avg",
        "demand_previous_avg", "demand_change_pct", "demand_trend", "demand_cv",
        "lead_time_mean", "lead_time_std", "lead_time_cv", "current_safety_stock",
        "recommended_safety_stock", "safety_stock_gap", "stockout_risk_score",
    ]
    return _columns(result.sort_values(sort_column, ascending=ascending).head(limit), columns), definitions


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
    limit = plan.limit
    facts: dict[str, Any] = {}
    definitions: dict[str, Any] = {}
    limitations = list(plan.limitations)
    table: pd.DataFrame | None = None
    direct_answer: str | None = None

    if plan.ambiguous_entities:
        direct_answer = "The entity reference is ambiguous. Please specify one of: " + "; ".join(plan.ambiguous_entities)
        limitations.append("No data query was executed because entity resolution was ambiguous.")
    elif plan.intent == "clarification":
        direct_answer = limitations[0] if limitations else "Please clarify the metric or entity you want analyzed."
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
            rows = _apply_single_domain_conditions(_filter_view(model.sku_view, plan), plan)
            if plan.intent == "supplier_demand":
                demand_sort = {
                    "demand_level": "demand_mean",
                    "demand_trend": "demand_change_pct",
                }.get(plan.metric, "demand_cv")
                table = _columns(rows.sort_values(
                    demand_sort, ascending=plan.direction == "ascending"
                ), [
                    "sku", "description", "supplier", "item_class", "demand_mean",
                    "demand_recent_avg", "demand_previous_avg", "demand_change_pct",
                    "demand_trend", "demand_std", "demand_cv",
                ]).head(limit)
                definitions["demand_cv"] = "demand standard deviation divided by mean demand"
            elif plan.intent == "supplier_receipts":
                receipt_sort = "lead_time_mean" if plan.metric == "lead_time_mean" else "lead_time_std"
                table = _columns(rows.sort_values(
                    receipt_sort, ascending=plan.direction == "ascending"
                ), [
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
        elif plan.metric == "lead_time_variability":
            summary = summary.sort_values(
                "receipt_level_lead_time_std_days", ascending=plan.direction == "ascending"
            )
        elif plan.metric == "lead_time_mean":
            summary = summary.sort_values(
                "receipt_weighted_average_lead_time_days", ascending=plan.direction == "ascending"
            )
        elif plan.metric == "stockout_risk":
            summary = summary.sort_values(
                "average_stockout_risk_score_across_skus", ascending=plan.direction == "ascending"
            )
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
        rows = _apply_single_domain_conditions(_filter_view(model.sku_view, plan), plan)
        if "increasing_demand" in plan.conditions or plan.metric == "demand_trend":
            rows = rows.loc[rows["demand_trend"] == "increasing"].sort_values(
                "demand_change_pct", ascending=plan.direction == "ascending"
            )
        else:
            demand_sort = "demand_mean" if plan.metric == "demand_level" else "demand_cv"
            rows = rows.sort_values(demand_sort, ascending=plan.direction == "ascending")
        table = _columns(rows, [
            "sku", "description", "supplier", "item_class", "demand_history_points",
            "demand_mean", "demand_std", "demand_cv", "demand_recent_avg",
            "demand_previous_avg", "demand_change_pct", "demand_trend",
        ]).head(limit)
        facts["demand_ranking"] = _records(table)
    elif plan.intent == "receipt_ranking":
        rows = _apply_single_domain_conditions(_filter_view(model.sku_view, plan), plan)
        if re.search(r"recent|trend|slower|faster|increas|decreas", q):
            limitations.append(
                "receipt_history has no order/receipt date; recent lead-time direction cannot be established."
            )
        sort_col = "lead_time_std" if plan.metric == "lead_time_variability" else "lead_time_mean"
        table = _columns(rows.sort_values(
            sort_col, ascending=plan.direction == "ascending"
        ), [
            "sku", "receipt_count",
            "lead_time_mean", "lead_time_median", "lead_time_std", "lead_time_cv",
            "lead_time_min", "lead_time_max", "lead_time_outlier_count",
        ]).head(limit)
        definitions["lead_time_variability"] = "sample standard deviation of actual lead-time days"
        facts["receipt_ranking"] = _records(table)
    elif plan.intent == "cross_file_risk":
        rows = _filter_view(model.sku_view, plan)
        table, cross_definitions = _cross_file_ranking(rows, plan, limit)
        definitions.update(cross_definitions)
        facts["cross_file_result"] = _records(table)
    elif plan.intent == "inventory_ranking":
        rows = _filter_view(model.sku_view, plan)
        sort_col = "rop_gap" if plan.metric == "rop_gap" else "safety_stock_gap"
        if "positive_safety_stock_gap" in plan.conditions:
            rows = rows.loc[rows["safety_stock_gap"] > 0]
        if "positive_rop_gap" in plan.conditions:
            rows = rows.loc[rows["rop_gap"] > 0]
        ascending = plan.direction == "ascending"
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
        "query_type": plan.query_type,
        "entity_type": plan.entity_type,
        "metric": plan.metric,
        "direction": plan.direction,
        "limit": plan.limit,
        "conditions": plan.conditions,
        "confidence": plan.confidence,
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
    interpretation_override: LanguageInterpretation | None = None,
) -> tuple[str, pd.DataFrame | None, dict[str, Any], str | None]:
    """Plan, execute, and serialize one question-specific evidence package."""
    model = data_model or InventoryDataModel.build(
        item_master, demand_history, receipt_history, analysis_results
    )
    plan = plan_question(question, model.item_master, prior_metadata, interpretation_override)
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


def _interpret_with_nemotron(
    question: str,
    *,
    client: Any,
    model: str,
) -> LanguageInterpretation | None:
    """Request a constrained plan only when deterministic confidence is low."""
    schema_prompt = """Interpret one inventory question. Return JSON only; no prose.
Allowed query_type: ranking, comparison, filter, metric_lookup, time_series,
explanation, forecast, profile, clarification.
Allowed entity_type: sku, supplier.
Allowed metric: lead_time_variability, lead_time_mean, demand_variability,
demand_level, demand_trend, safety_stock_gap, rop_gap, stockout_risk, or null.
Allowed direction: ascending, descending, or null.
Allowed conditions: high_demand_variability, high_lead_time_variability,
increasing_demand, stable_demand, stable_lead_time, long_lead_time,
positive_safety_stock_gap, positive_rop_gap, low_current_safety_stock.
Use keys: query_type, entity_type, metric, metrics, direction, limit, conditions.
Do not calculate values, name results, write code, or add fields."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": schema_prompt},
                {"role": "user", "content": question[:1000]},
            ],
            temperature=0,
            max_tokens=250,
        )
        content = str(response.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            return None
        payload = json.loads(match.group(0))
        return validate_interpretation_payload(payload, maximum_rows=MAX_CONTEXT_ROWS)
    except Exception:
        # Parsing is an optional fallback. A failed/invalid interpretation is
        # safer as a clarification than as a forced analytical operation.
        return None


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
    config: dict[str, str] | None = None
    interpretation_override: LanguageInterpretation | None = None
    preliminary_plan = plan_question(question, item_master, prior_metadata)
    if preliminary_plan.intent == "clarification" and not preliminary_plan.ambiguous_entities:
        try:
            config = get_nvidia_config(api_key, model, base_url)
            if client is None:
                from openai import OpenAI
                client = OpenAI(
                    base_url=config["base_url"], api_key=config["api_key"], timeout=30.0, max_retries=1
                )
            interpretation_override = _interpret_with_nemotron(
                question, client=client, model=config["model"]
            )
        except (MissingNvidiaAPIKey, ImportError):
            interpretation_override = None
    try:
        context, table, metadata, direct_answer = build_analytical_context(
            question, item_master, demand_history, receipt_history, analysis_results,
            prior_metadata, data_model, interpretation_override,
        )
    except InventoryDataError as exc:
        raise InventoryAssistantError(str(exc)) from exc
    evidence = metadata.get("evidence", {})
    if direct_answer:
        return AssistantAnswer(direct_answer, table, metadata, evidence)

    config = config or get_nvidia_config(api_key, model, base_url)
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

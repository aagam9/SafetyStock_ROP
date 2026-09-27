"""Grounded analytics and NVIDIA Nemotron integration for the inventory assistant.

Numerical work stays in this module's deterministic pandas/numpy helpers.  The
language model receives only a compact question-specific evidence packet and is
asked to explain it; it never receives the complete uploaded data sets.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")
MAX_HISTORY_MESSAGES = 6
MAX_CONTEXT_ROWS = 15
SESSION_ANALYSIS_KEYS = (
    "last_report", "last_summary", "item_master", "demand_history",
    "receipt_history", "analysis_results", "review_queue", "analysis_summary",
    "inventory_chat", "assistant_metadata", "last_input_signature",
)

SYSTEM_PROMPT = """You are an inventory analytics assistant working only with the
evidence supplied by the application. Do not invent SKUs, suppliers, demand,
stockouts, lead times, calculations, or business events. Never imply that you
calculated a number: Python calculated every number in the evidence. Clearly
separate data facts from interpretation, acknowledge unavailable information,
and do not treat estimated Stockout Risk as historical stockout evidence.
Supplier evidence currently measures lead-time performance and consistency,
not OTIF, fill rate, quality, or promised-date adherence. Answer the user's
question directly, cite the most relevant supplied numbers, explain the
planning implication, and keep the response practical and concise."""


class InventoryAssistantError(RuntimeError):
    """Safe, user-facing error from assistant configuration or execution."""


class MissingNvidiaAPIKey(InventoryAssistantError):
    """Raised when only the optional assistant is missing its credential."""


@dataclass
class AssistantAnswer:
    text: str
    table: pd.DataFrame | None
    metadata: dict[str, Any]


def reset_inventory_session(state: Any) -> None:
    """Remove data-dependent session values while leaving unrelated UI state intact."""
    for key in SESSION_ANALYSIS_KEYS:
        state.pop(key, None)


def get_nvidia_config(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, str]:
    """Resolve centralized NVIDIA configuration without exposing the key."""
    resolved_key = (api_key or os.getenv("NVIDIA_API_KEY", "")).strip()
    if not resolved_key:
        raise MissingNvidiaAPIKey(
            "Inventory analysis is available, but the AI assistant requires an NVIDIA API key."
        )
    return {
        "api_key": resolved_key,
        "model": (model or os.getenv("NVIDIA_MODEL") or NVIDIA_MODEL).strip(),
        "base_url": (base_url or os.getenv("NVIDIA_BASE_URL") or NVIDIA_BASE_URL).rstrip("/"),
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


def get_sku_summary(
    analysis_results: Sequence[Mapping[str, Any]] | pd.DataFrame, sku: str
) -> dict[str, Any] | None:
    frame = _results_frame(analysis_results)
    match = frame.loc[frame["sku"].astype(str).str.casefold() == str(sku).strip().casefold()]
    return None if match.empty else match.iloc[0].to_dict()


def forecast_demand(demand_history: pd.DataFrame, sku: str, horizon: int = 4) -> dict[str, Any]:
    """Return a small, deterministic damped-trend weekly forecast.

    A least-squares line is fitted to at most the latest 13 observations. The
    fitted trend is damped by 50% to avoid extending a short-term slope at full
    strength. Forecast values are floored at zero.
    """
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
    prediction = np.maximum(0.0, fitted_latest + (0.5 * slope * future))
    forecast = pd.DataFrame(
        {
            "sku": [sku] * horizon,
            "forecast_period": np.arange(1, horizon + 1),
            "forecast_units": np.round(prediction, 1),
        }
    )
    return {
        "sku": sku,
        "horizon": horizon,
        "method": "13-period linear trend with 50% trend damping",
        "available_observations": int(len(values)),
        "observations_used": int(len(recent)),
        "recent_average": round(float(recent.mean()), 1),
        "recent_trend_units_per_period": round(float(slope), 2),
        "forecast_total": round(float(prediction.sum()), 1),
        "forecast": forecast,
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
    frame["safety_stock_gap"] = (
        frame["recommended_safety_stock"] - frame["current_safety_stock"]
    )
    if item_class:
        frame = frame.loc[frame["item_class"].astype(str).str.upper() == item_class.upper()]
    if supplier:
        frame = frame.loc[frame["supplier"].astype(str).str.casefold() == supplier.casefold()]
    ascending = direction == "decrease"
    frame = frame.loc[
        frame["safety_stock_gap"] < 0 if ascending else frame["safety_stock_gap"] > 0
    ]
    columns = [
        "sku", "description", "supplier", "item_class", "current_safety_stock",
        "recommended_safety_stock", "safety_stock_gap", "recent_weekly_std",
        "lt_actual_std_days",
    ]
    return frame.sort_values("safety_stock_gap", ascending=ascending).head(limit)[columns]


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
    columns = [
        "sku", "description", "supplier", "item_class", "current_rop",
        "recommended_rop", "rop_gap",
    ]
    return frame.head(limit)[columns]


def calculate_stockout_risk(
    analysis_results: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Create a transparent relative planning-risk score (not stockout history)."""
    frame = _results_frame(analysis_results).copy()
    rec_ss = frame["recommended_safety_stock"].replace(0, np.nan)
    rec_rop = frame["recommended_rop"].replace(0, np.nan)
    mean_demand = frame["recent_weekly_mean"].replace(0, np.nan)
    mean_lt = frame["lt_actual_mean_days"].replace(0, np.nan)
    ss_gap = ((frame["recommended_safety_stock"] - frame["current_safety_stock"]) / rec_ss).clip(0, 1)
    rop_gap = ((frame["recommended_rop"] - frame["current_rop"]) / rec_rop).clip(0, 1)
    demand_cv = (frame["recent_weekly_std"] / mean_demand).clip(0, 2) / 2
    lead_time_cv = (frame["lt_actual_std_days"] / mean_lt).clip(0, 1)
    positive_trend = (frame["trend_pct"].clip(lower=0, upper=100) / 100)
    class_weight = frame["item_class"].map({"A": 1.0, "B": 0.6, "C": 0.3}).fillna(0.6)
    score = 100 * (
        0.30 * ss_gap.fillna(0)
        + 0.20 * rop_gap.fillna(0)
        + 0.15 * demand_cv.fillna(0)
        + 0.15 * lead_time_cv.fillna(0)
        + 0.10 * positive_trend.fillna(0)
        + 0.10 * class_weight
    )
    frame["safety_stock_gap"] = frame["recommended_safety_stock"] - frame["current_safety_stock"]
    frame["rop_gap"] = frame["recommended_rop"] - frame["current_rop"]
    frame["stockout_risk_score"] = score.round(1)
    frame["stockout_risk_level"] = pd.cut(
        frame["stockout_risk_score"],
        bins=[-np.inf, 25, 50, np.inf],
        labels=["Low", "Moderate", "High"],
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
    receipt_supplier = receipt_history.merge(
        item_master[["sku", "supplier"]], on="sku", how="left", validate="many_to_one"
    )
    receipt_stats = receipt_supplier.groupby("supplier", dropna=False).agg(
        sku_count=("sku", "nunique"),
        receipt_count=("actual_lead_time_days", "size"),
        average_lead_time_days=("actual_lead_time_days", "mean"),
        median_lead_time_days=("actual_lead_time_days", "median"),
        minimum_lead_time_days=("actual_lead_time_days", "min"),
        maximum_lead_time_days=("actual_lead_time_days", "max"),
        lead_time_std_days=("actual_lead_time_days", "std"),
    )
    receipt_stats["lead_time_std_days"] = receipt_stats["lead_time_std_days"].fillna(0)
    receipt_stats["lead_time_cv"] = (
        receipt_stats["lead_time_std_days"] / receipt_stats["average_lead_time_days"].replace(0, np.nan)
    )

    risk = calculate_stockout_risk(analysis_results)
    results = _results_frame(analysis_results).copy()
    results["safety_stock_gap"] = results["recommended_safety_stock"] - results["current_safety_stock"]
    results["rop_gap"] = results["recommended_rop"] - results["current_rop"]
    results = results.merge(risk[["sku", "stockout_risk_score"]], on="sku", how="left")
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
    wanted = {s.casefold() for s in skus}
    frame = frame.loc[frame["sku"].astype(str).str.casefold().isin(wanted)]
    desired_columns = [
        "sku", "description", "supplier", "item_class", "recent_weekly_mean",
        "recent_weekly_std", "trend_pct", "zero_demand_share", "is_intermittent",
        "has_outliers", "assumed_lead_time_days", "lt_actual_mean_days",
        "lt_actual_std_days", "lt_drift_pct", "service_level_z",
        "current_safety_stock", "recommended_safety_stock", "current_rop", "recommended_rop",
        "ss_drift_pct", "analysis_flags",
    ]
    return frame[[column for column in desired_columns if column in frame.columns]]


def compare_suppliers(supplier_summary: pd.DataFrame, suppliers: Sequence[str]) -> pd.DataFrame:
    wanted = {s.casefold() for s in suppliers}
    return supplier_summary.loc[
        supplier_summary["supplier"].astype(str).str.casefold().isin(wanted)
    ].copy()


def detect_intent(question: str, prior_intent: str | None = None) -> str:
    q = question.casefold()
    if re.search(r"\b(otif|fill rate|quality|promised[- ]date|on[- ]time delivery)\b", q):
        return "unsupported"
    historical = re.search(r"\b(how many|history|historical|last year|occurred|events?)\b", q)
    if "stockout" in q or "stock out" in q:
        if historical and "risk" not in q:
            return "unsupported"
        return "stockout_risk"
    if re.search(r"affected by supplier delays?|combin\w* high demand.*lead[ -]?time", q):
        return "stockout_risk"
    if re.search(r"\b(forecast|predict|projection|next \d+ (?:weeks?|periods?))\b", q):
        return "forecast"
    if re.search(r"\bsuppliers?\s+contribut|\bsuppliers?\s+create", q):
        return "supplier_performance"
    if re.search(r"\b(safety stock|\bss\b|under[- ]protected|over[- ]protected)\b", q):
        return "safety_stock"
    if re.search(r"\b(reorder point|\brop\b)\b", q):
        return "reorder_point"
    if re.search(r"\bwhich suppliers?\b|\bsupplier\s+has\b", q):
        return "supplier_performance"
    if "compare" in q and "supplier" in q:
        return "supplier_comparison"
    if "compare" in q:
        return "sku_comparison"
    if re.search(r"\b(supplier|vendor)\b", q):
        return "supplier_performance"
    if re.search(r"\b(receipt|shipment|lead[ -]?time|purchase order|\bpo\b)\b", q):
        return "receipt_performance"
    if re.search(r"\b(demand|volatile|variability|trend|intermittent|declining|growing)\b", q):
        return "demand"
    if q.strip() in {"why", "why?", "explain", "tell me more", "what about it?"} and prior_intent:
        return prior_intent
    return "general_inventory"


def _extract_entities(question: str, values: Sequence[Any]) -> list[str]:
    q = question.casefold()
    matches = [
        str(value)
        for value in values
        if re.search(
            rf"(?<![a-z0-9]){re.escape(str(value).casefold())}(?![a-z0-9])",
            q,
        )
    ]
    return sorted(set(matches), key=lambda value: (-len(value), value))


def _limit_from_question(question: str, default: int = 10) -> int:
    match = re.search(r"\btop\s+(\d{1,2})\b", question.casefold())
    return min(max(int(match.group(1)), 1), MAX_CONTEXT_ROWS) if match else default


def _records(frame: pd.DataFrame, limit: int = MAX_CONTEXT_ROWS) -> list[dict[str, Any]]:
    clean = frame.head(limit).astype(object).where(pd.notna(frame.head(limit)), None)
    return json.loads(clean.to_json(orient="records"))


def build_analytical_context(
    question: str,
    item_master: pd.DataFrame,
    demand_history: pd.DataFrame,
    receipt_history: pd.DataFrame,
    analysis_results: Sequence[Mapping[str, Any]] | pd.DataFrame,
    prior_metadata: Mapping[str, Any] | None = None,
) -> tuple[str, pd.DataFrame | None, dict[str, Any], str | None]:
    """Retrieve/calculate evidence and return compact context plus an optional table."""
    results = _results_frame(analysis_results)
    prior_metadata = dict(prior_metadata or {})
    intent = detect_intent(question, prior_metadata.get("intent"))
    skus = _extract_entities(question, results["sku"].astype(str).tolist())
    suppliers = _extract_entities(question, item_master["supplier"].astype(str).unique().tolist())
    if not skus:
        skus = list(prior_metadata.get("skus", []))
    if not suppliers:
        suppliers = list(prior_metadata.get("suppliers", []))
    limit = _limit_from_question(question)
    q = question.casefold()
    facts: dict[str, Any] = {
        "intent": intent,
        "evidence_scope": {
            "sku_count": int(results["sku"].nunique()),
            "demand_rows": int(len(demand_history)),
            "receipt_rows": int(len(receipt_history)),
        },
    }
    table: pd.DataFrame | None = None
    direct_answer: str | None = None

    if intent == "unsupported":
        if "stockout" in q or "stock out" in q:
            direct_answer = (
                "Historical stockout events are not available in the uploaded datasets. "
                "I can evaluate Stockout Risk using demand variability, lead-time variability, "
                "and current versus recommended Safety Stock and Reorder Point."
            )
        else:
            direct_answer = (
                "The uploaded data cannot establish OTIF, fill rate, supplier quality, or "
                "promised-date adherence. Current supplier analysis is limited to actual "
                "lead-time performance and consistency."
            )
        facts["limitation"] = direct_answer
    elif intent == "forecast":
        if not skus:
            direct_answer = "Please specify a SKU for the demand forecast."
        else:
            horizon_match = re.search(r"\bnext\s+(\d{1,2})\s+(?:weeks?|periods?)\b", q)
            horizon = int(horizon_match.group(1)) if horizon_match else 4
            forecast = forecast_demand(demand_history, skus[0], horizon)
            table = forecast.pop("forecast")
            facts["forecast"] = forecast
    elif intent == "sku_comparison":
        if len(skus) < 2:
            direct_answer = "Please name at least two SKUs to compare."
        else:
            table = compare_skus(results, skus[:5])
            facts["sku_comparison"] = _records(table)
    elif intent in {"supplier_performance", "supplier_comparison"}:
        summary = get_supplier_summary(item_master, receipt_history, results)
        if intent == "supplier_comparison":
            if len(suppliers) < 2:
                direct_answer = "Please name at least two suppliers to compare."
            else:
                table = compare_suppliers(summary, suppliers[:5])
        elif suppliers:
            supplier = suppliers[0]
            if "what sku" in q or "which sku" in q or "products" in q or "items" in q:
                table = results.loc[
                    results["supplier"].astype(str).str.casefold() == supplier.casefold(),
                    ["sku", "description", "item_class", "current_safety_stock", "recommended_safety_stock"],
                ].head(limit)
            else:
                table = compare_suppliers(summary, [supplier])
        else:
            if "risk" in q:
                sort_column = "average_stockout_risk_score"
            elif re.search(r"contribut|requirement|safety stock", q):
                sort_column = "total_recommended_safety_stock"
            else:
                sort_column = "lead_time_cv" if re.search(r"consistent|variab", q) else "average_lead_time_days"
            ascending = "most consistent" in q
            table = summary.sort_values(sort_column, ascending=ascending).head(limit)
        if table is not None:
            facts["supplier_lead_time_summary"] = _records(table)
            facts["supplier_metric_limitation"] = (
                "Supplier metrics describe lead-time performance and consistency, not full supplier performance."
            )
    elif intent == "receipt_performance":
        receipt = receipt_history.merge(
            item_master[["sku", "description", "supplier"]],
            on="sku", how="left", validate="many_to_one",
        )
        if skus:
            receipt = receipt.loc[receipt["sku"].astype(str).str.casefold() == skus[0].casefold()]
        if re.search(r"unusually|outlier|longest receipts?", q):
            receipt = receipt.copy()
            group = receipt.groupby("sku")["actual_lead_time_days"]
            receipt["sku_lead_time_mean"] = group.transform("mean")
            receipt["sku_lead_time_std"] = group.transform("std").fillna(0)
            receipt["lead_time_z_score"] = (
                (receipt["actual_lead_time_days"] - receipt["sku_lead_time_mean"])
                / receipt["sku_lead_time_std"].replace(0, np.nan)
            ).fillna(0)
            table = receipt.sort_values(
                ["lead_time_z_score", "actual_lead_time_days"], ascending=False
            ).head(limit)[[
                "sku", "description", "supplier", "po_number", "actual_lead_time_days",
                "sku_lead_time_mean", "lead_time_z_score",
            ]].round(2)
            facts["unusually_long_receipts"] = _records(table)
            metadata = {"intent": intent, "skus": skus[:5], "suppliers": suppliers[:5]}
            context = "QUESTION\n" + question + "\n\nPYTHON-GENERATED EVIDENCE\n" + json.dumps(
                facts, indent=2, default=str
            )
            return context, table, metadata, direct_answer
        grouped = receipt.groupby(["sku", "description", "supplier"], as_index=False).agg(
            receipt_count=("actual_lead_time_days", "size"),
            average_lead_time_days=("actual_lead_time_days", "mean"),
            median_lead_time_days=("actual_lead_time_days", "median"),
            minimum_lead_time_days=("actual_lead_time_days", "min"),
            maximum_lead_time_days=("actual_lead_time_days", "max"),
            lead_time_std_days=("actual_lead_time_days", "std"),
            lead_time_trend_days_per_receipt=(
                "actual_lead_time_days",
                lambda values: float(np.polyfit(np.arange(len(values)), values, 1)[0])
                if len(values) > 1 else 0.0,
            ),
        ).fillna({"lead_time_std_days": 0})
        if re.search(r"increas|trend", q):
            sort_col = "lead_time_trend_days_per_receipt"
        else:
            sort_col = "lead_time_std_days" if re.search(r"inconsistent|variab", q) else "average_lead_time_days"
        table = grouped.sort_values(sort_col, ascending=False).head(limit).round(2)
        facts["receipt_performance"] = _records(table)
    elif intent == "safety_stock":
        if skus:
            table = compare_skus(results, skus[:5])
        else:
            direction = "decrease" if re.search(r"excess|decrease|reduce|above", q) else "increase"
            item_class = "A" if re.search(r"\b(class\s+)?a\s+(items?|skus?|products?)\b", q) else None
            table = rank_safety_stock_gaps(
                results, limit, direction, item_class, suppliers[0] if suppliers else None
            )
            if re.search(r"increas\w* demand|demand.*increas", q):
                trending_skus = set(results.loc[results["trend_pct"] > 0, "sku"])
                table = table.loc[table["sku"].isin(trending_skus)].head(limit)
        facts["safety_stock_evidence"] = _records(table)
    elif intent == "reorder_point":
        if skus:
            table = compare_skus(results, skus[:5])
        else:
            direction = "decrease" if re.search(r"decrease|reduce|lower", q) else (
                "increase" if re.search(r"increase|raise|higher", q) else "absolute"
            )
            table = rank_rop_gaps(results, limit, direction, suppliers[0] if suppliers else None)
        facts["reorder_point_evidence"] = _records(table)
    elif intent == "stockout_risk":
        risk = calculate_stockout_risk(results)
        if skus:
            risk = risk.loc[risk["sku"].astype(str).str.casefold().isin({s.casefold() for s in skus})]
        if suppliers:
            risk = risk.loc[risk["supplier"].astype(str).str.casefold() == suppliers[0].casefold()]
        if re.search(r"\b(class\s+)?a\s+(items?|skus?|products?)\b", q):
            risk = risk.loc[risk["item_class"] == "A"]
        table = risk.head(limit)
        facts["stockout_risk"] = _records(table)
        facts["risk_definition"] = (
            "Relative 0-100 planning-risk score: 30% SS gap, 20% ROP gap, 15% demand "
            "variability, 15% lead-time variability, 10% positive demand trend, 10% ABC criticality."
        )
    elif intent == "demand":
        demand_results = results
        if re.search(r"\b(class\s+)?a\s+(items?|skus?|products?)\b", q):
            demand_results = demand_results.loc[demand_results["item_class"] == "A"]
        if skus:
            table = compare_skus(demand_results, skus[:5])
        elif "intermittent" in q:
            table = demand_results.sort_values("zero_demand_share", ascending=False).head(limit)[[
                "sku", "description", "supplier", "item_class", "zero_demand_share", "is_intermittent"
            ]]
        elif "declin" in q:
            table = demand_results.sort_values("trend_pct").head(limit)[[
                "sku", "description", "supplier", "item_class", "recent_weekly_mean", "trend_pct"
            ]]
        elif re.search(r"increas|grow", q):
            table = demand_results.sort_values("trend_pct", ascending=False).head(limit)[[
                "sku", "description", "supplier", "item_class", "recent_weekly_mean", "trend_pct"
            ]]
        else:
            table = demand_results.sort_values("recent_weekly_std", ascending=False).head(limit)[[
                "sku", "description", "supplier", "item_class", "recent_weekly_mean",
                "recent_weekly_std", "trend_pct", "zero_demand_share"
            ]]
        facts["demand_evidence"] = _records(table)
    else:
        risk = calculate_stockout_risk(results)
        if skus:
            risk = risk.loc[risk["sku"].astype(str).str.casefold().isin({s.casefold() for s in skus})]
        if suppliers:
            risk = risk.loc[risk["supplier"].astype(str).str.casefold() == suppliers[0].casefold()]
        table = risk.head(limit)
        facts["portfolio_risk_overview"] = _records(table)

    metadata = {"intent": intent, "skus": skus[:5], "suppliers": suppliers[:5]}
    context = "QUESTION\n" + question + "\n\nPYTHON-GENERATED EVIDENCE\n" + json.dumps(
        facts, indent=2, default=str
    )
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
) -> AssistantAnswer:
    """Answer one grounded question, using Nemotron only for explanation."""
    if not question or not question.strip():
        raise ValueError("Enter a question about the completed inventory analysis.")
    context, table, metadata, direct_answer = build_analytical_context(
        question, item_master, demand_history, receipt_history, analysis_results, prior_metadata
    )
    if direct_answer:
        return AssistantAnswer(direct_answer, table, metadata)

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
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": str(content)[:2000]})
    messages.append(
        {
            "role": "user",
            "content": context
            + "\n\nRespond using only this evidence. The application displays any evidence table separately.",
        }
    )
    try:
        response = client.chat.completions.create(
            model=config["model"], messages=messages, temperature=0.2, max_tokens=700
        )
        text = response.choices[0].message.content
        if not text or not str(text).strip():
            raise InventoryAssistantError("NVIDIA returned an empty response. Try again.")
    except InventoryAssistantError:
        raise
    except Exception as exc:
        raise _safe_api_error(exc) from exc
    return AssistantAnswer(str(text).strip(), table, metadata)

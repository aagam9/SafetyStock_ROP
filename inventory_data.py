"""Relationship-aware data model for the inventory analytics assistant.

The item master is the SKU dimension. Demand and receipt histories are separate
fact tables and are always aggregated independently to SKU grain before they
are combined. This module intentionally contains no LLM or Streamlit logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_DEMAND_WINDOW = 8
DEMAND_SUMMARY_COLUMNS = [
    "sku", "demand_history_points", "demand_total", "demand_mean", "demand_median",
    "demand_std", "demand_cv", "demand_zero_periods", "demand_zero_share",
    "demand_recent_window", "demand_recent_avg", "demand_previous_window",
    "demand_previous_avg", "demand_change_pct", "demand_trend",
]
RECEIPT_SUMMARY_COLUMNS = [
    "sku", "receipt_count", "lead_time_mean", "lead_time_median", "lead_time_std",
    "lead_time_cv", "lead_time_min", "lead_time_max", "lead_time_outlier_count",
    "receipt_chronology_available",
]


class InventoryDataError(ValueError):
    """Raised when the relationship model cannot be built unambiguously."""


def _records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    selected = frame if limit is None else frame.head(limit)
    clean = selected.astype(object).where(pd.notna(selected), None)
    return clean.to_dict(orient="records")


def _safe_pct_change(current: float, previous: float) -> float | None:
    if pd.isna(previous) or previous == 0:
        return None
    return (current - previous) / previous * 100


def _trend_label(recent: float, previous: float, change_pct: float | None) -> str:
    if pd.isna(previous):
        return "insufficient_history"
    if previous == 0:
        return "increasing" if recent > 0 else "stable"
    if change_pct is not None and change_pct > 10:
        return "increasing"
    if change_pct is not None and change_pct < -10:
        return "decreasing"
    return "stable"


def _prepare_demand_history(demand_history: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    required = {"sku", "week", "demand_qty"}
    missing = required - set(demand_history.columns)
    if missing:
        raise InventoryDataError(
            "demand_history is missing required column(s): " + ", ".join(sorted(missing))
        )
    frame = demand_history.copy()
    frame["sku"] = frame["sku"].astype(str).str.strip()
    frame["demand_qty"] = pd.to_numeric(frame["demand_qty"], errors="coerce")
    frame["_source_order"] = np.arange(len(frame))
    numeric_period = pd.to_numeric(frame["week"], errors="coerce")
    if numeric_period.notna().all():
        frame["_period_sort"] = numeric_period
        period_kind = "numeric_week"
    else:
        date_period = pd.to_datetime(frame["week"], errors="coerce", format="mixed")
        if date_period.notna().all():
            frame["_period_sort"] = date_period
            period_kind = "date"
        else:
            frame["_period_sort"] = frame["_source_order"]
            period_kind = "source_order"
    frame = frame.sort_values(["sku", "_period_sort", "_source_order"], kind="stable")
    return frame, period_kind


def aggregate_demand_history(
    demand_history: pd.DataFrame, recent_window: int = DEFAULT_DEMAND_WINDOW
) -> pd.DataFrame:
    """Aggregate demand independently to one row per SKU."""
    frame, _ = _prepare_demand_history(demand_history)
    rows: list[dict[str, Any]] = []
    for sku, group in frame.groupby("sku", sort=False):
        values = group["demand_qty"].dropna().to_numpy(dtype=float)
        recent = values[-min(recent_window, len(values)) :] if len(values) else values
        previous_end = max(len(values) - len(recent), 0)
        previous_start = max(previous_end - recent_window, 0)
        previous = values[previous_start:previous_end]
        mean = float(values.mean()) if len(values) else np.nan
        std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        recent_avg = float(recent.mean()) if len(recent) else np.nan
        previous_avg = float(previous.mean()) if len(previous) else np.nan
        change_pct = _safe_pct_change(recent_avg, previous_avg)
        rows.append(
            {
                "sku": sku,
                "demand_history_points": int(len(values)),
                "demand_total": float(values.sum()) if len(values) else np.nan,
                "demand_mean": mean if pd.notna(mean) else np.nan,
                "demand_median": float(np.median(values)) if len(values) else np.nan,
                "demand_std": std if pd.notna(std) else np.nan,
                "demand_cv": std / mean if pd.notna(mean) and mean != 0 else np.nan,
                "demand_zero_periods": int((values == 0).sum()),
                "demand_zero_share": float((values == 0).mean()) if len(values) else np.nan,
                "demand_recent_window": int(len(recent)),
                "demand_recent_avg": recent_avg if pd.notna(recent_avg) else np.nan,
                "demand_previous_window": int(len(previous)),
                "demand_previous_avg": previous_avg if pd.notna(previous_avg) else np.nan,
                "demand_change_pct": change_pct,
                "demand_trend": _trend_label(recent_avg, previous_avg, change_pct),
            }
        )
    return pd.DataFrame(rows, columns=DEMAND_SUMMARY_COLUMNS)


def aggregate_receipt_history(receipt_history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate receipts independently to one row per SKU.

    The current receipt schema has no chronological field, so no recent/past
    receipt comparison or lead-time trend is calculated.
    """
    required = {"sku", "po_number", "actual_lead_time_days"}
    missing = required - set(receipt_history.columns)
    if missing:
        raise InventoryDataError(
            "receipt_history is missing required column(s): " + ", ".join(sorted(missing))
        )
    frame = receipt_history.copy()
    frame["sku"] = frame["sku"].astype(str).str.strip()
    frame["actual_lead_time_days"] = pd.to_numeric(
        frame["actual_lead_time_days"], errors="coerce"
    )
    rows: list[dict[str, Any]] = []
    for sku, group in frame.groupby("sku", sort=False):
        values = group["actual_lead_time_days"].dropna().to_numpy(dtype=float)
        mean = float(values.mean()) if len(values) else np.nan
        std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        if len(values) >= 4:
            q1, q3 = np.percentile(values, [25, 75])
            upper = q3 + 1.5 * (q3 - q1)
            outlier_count = int((values > upper).sum())
        else:
            outlier_count = 0
        rows.append(
            {
                "sku": sku,
                "receipt_count": int(len(values)),
                "lead_time_mean": mean if pd.notna(mean) else np.nan,
                "lead_time_median": float(np.median(values)) if len(values) else np.nan,
                "lead_time_std": std if pd.notna(std) else np.nan,
                "lead_time_cv": std / mean if pd.notna(mean) and mean != 0 else np.nan,
                "lead_time_min": float(values.min()) if len(values) else np.nan,
                "lead_time_max": float(values.max()) if len(values) else np.nan,
                "lead_time_outlier_count": outlier_count,
                "receipt_chronology_available": False,
            }
        )
    return pd.DataFrame(rows, columns=RECEIPT_SUMMARY_COLUMNS)


def _analysis_frame(
    analysis_results: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    frame = (
        analysis_results.copy()
        if isinstance(analysis_results, pd.DataFrame)
        else pd.DataFrame(list(analysis_results))
    )
    if frame.empty:
        return pd.DataFrame(columns=["sku"])
    if "sku" not in frame.columns:
        raise InventoryDataError("analysis_results is missing the SKU column.")
    frame["sku"] = frame["sku"].astype(str).str.strip()
    if frame["sku"].duplicated().any():
        duplicates = sorted(frame.loc[frame["sku"].duplicated(False), "sku"].unique())
        raise InventoryDataError(
            "analysis_results must contain at most one row per SKU. Duplicate SKU(s): "
            + ", ".join(duplicates[:10])
        )
    return frame


def _add_relationship_risk(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rec_ss = pd.to_numeric(result.get("recommended_safety_stock"), errors="coerce").replace(0, np.nan)
    rec_rop = pd.to_numeric(result.get("recommended_rop"), errors="coerce").replace(0, np.nan)
    ss_gap_ratio = (result["safety_stock_gap"] / rec_ss).clip(0, 1).fillna(0)
    rop_gap_ratio = (result["rop_gap"] / rec_rop).clip(0, 1).fillna(0)
    demand_cv = pd.to_numeric(result.get("demand_cv"), errors="coerce").clip(0, 2).fillna(0) / 2
    lead_cv = pd.to_numeric(result.get("lead_time_cv"), errors="coerce").clip(0, 1).fillna(0)
    trend = pd.to_numeric(result.get("demand_change_pct"), errors="coerce").clip(0, 100).fillna(0) / 100
    criticality = result["item_class"].astype(str).str.upper().map(
        {"A": 1.0, "B": 0.6, "C": 0.3}
    ).fillna(0.6)
    result["stockout_risk_score"] = (
        100
        * (
            0.30 * ss_gap_ratio
            + 0.20 * rop_gap_ratio
            + 0.15 * demand_cv
            + 0.15 * lead_cv
            + 0.10 * trend
            + 0.10 * criticality
        )
    ).round(1)
    result["stockout_risk_level"] = pd.cut(
        result["stockout_risk_score"],
        bins=[-np.inf, 25, 50, np.inf],
        labels=["Low", "Moderate", "High"],
    ).astype(str)
    return result


def _present(value: Any) -> bool:
    if isinstance(value, (list, tuple, dict, set)):
        return True
    marker = pd.notna(value)
    return bool(marker) if not isinstance(marker, (np.ndarray, pd.Series)) else bool(marker.all())


@dataclass
class InventoryDataModel:
    """Validated, cached relationship view for one completed analysis session."""

    item_master: pd.DataFrame
    demand_history: pd.DataFrame
    receipt_history: pd.DataFrame
    analysis_results: pd.DataFrame
    demand_summary: pd.DataFrame
    receipt_summary: pd.DataFrame
    sku_view: pd.DataFrame
    demand_period_kind: str

    @classmethod
    def build(
        cls,
        item_master: pd.DataFrame,
        demand_history: pd.DataFrame,
        receipt_history: pd.DataFrame,
        analysis_results: Sequence[Mapping[str, Any]] | pd.DataFrame,
        recent_window: int = DEFAULT_DEMAND_WINDOW,
    ) -> "InventoryDataModel":
        items = item_master.copy()
        required_items = {"sku", "description", "supplier", "item_class"}
        missing = required_items - set(items.columns)
        if missing:
            raise InventoryDataError(
                "item_master is missing required column(s): " + ", ".join(sorted(missing))
            )
        items["sku"] = items["sku"].astype(str).str.strip()
        items["supplier"] = items["supplier"].astype(str).str.strip()
        if items["sku"].duplicated().any():
            duplicates = sorted(items.loc[items["sku"].duplicated(False), "sku"].unique())
            raise InventoryDataError(
                "item_master must contain exactly one row per SKU. Duplicate SKU(s): "
                + ", ".join(duplicates[:10])
            )

        prepared_demand, period_kind = _prepare_demand_history(demand_history)
        demand_summary = aggregate_demand_history(prepared_demand, recent_window)
        receipt_summary = aggregate_receipt_history(receipt_history)
        analysis = _analysis_frame(analysis_results)

        # Every merge is dimension-to-one-row-per-SKU with explicit validation.
        # Demand history and receipt history are never merged directly.
        view = items.merge(demand_summary, on="sku", how="left", validate="one_to_one")
        view = view.merge(receipt_summary, on="sku", how="left", validate="one_to_one")
        analysis_columns = [
            "sku", "recommended_safety_stock", "recommended_rop", "ss_drift_pct",
            "lt_drift_pct", "service_level_z", "flagged", "analysis_flags",
            "dollar_impact", "action", "confidence", "risk_if_ignored", "rationale",
        ]
        available_analysis = [column for column in analysis_columns if column in analysis.columns]
        view = view.merge(
            analysis[available_analysis], on="sku", how="left", validate="one_to_one"
        )
        for column in (
            "current_safety_stock", "recommended_safety_stock", "current_rop", "recommended_rop"
        ):
            if column not in view.columns:
                view[column] = np.nan
            view[column] = pd.to_numeric(view[column], errors="coerce")
        view["safety_stock_gap"] = (
            view["recommended_safety_stock"] - view["current_safety_stock"]
        )
        view["rop_gap"] = view["recommended_rop"] - view["current_rop"]
        view["safety_stock_gap_pct"] = (
            view["safety_stock_gap"] / view["current_safety_stock"].replace(0, np.nan) * 100
        ).round(2)
        view["rop_gap_pct"] = (
            view["rop_gap"] / view["current_rop"].replace(0, np.nan) * 100
        ).round(2)
        view = _add_relationship_risk(view)

        prepared_receipts = receipt_history.copy()
        if "receipt_row_id" not in prepared_receipts.columns:
            prepared_receipts = prepared_receipts.reset_index(drop=True)
            prepared_receipts["receipt_row_id"] = [
                f"receipt-{index + 1}" for index in range(len(prepared_receipts))
            ]
        prepared_receipts["sku"] = prepared_receipts["sku"].astype(str).str.strip()
        prepared_receipts["actual_lead_time_days"] = pd.to_numeric(
            prepared_receipts["actual_lead_time_days"], errors="coerce"
        )
        return cls(
            item_master=items,
            demand_history=prepared_demand,
            receipt_history=prepared_receipts,
            analysis_results=analysis,
            demand_summary=demand_summary,
            receipt_summary=receipt_summary,
            sku_view=view,
            demand_period_kind=period_kind,
        )

    def _canonical_sku(self, sku: str) -> str | None:
        match = self.item_master.loc[
            self.item_master["sku"].astype(str).str.casefold() == str(sku).strip().casefold(), "sku"
        ]
        return None if match.empty else str(match.iloc[0])

    def _canonical_supplier(self, supplier: str) -> str | None:
        match = self.item_master.loc[
            self.item_master["supplier"].astype(str).str.casefold()
            == str(supplier).strip().casefold(),
            "supplier",
        ]
        return None if match.empty else str(match.iloc[0])

    def demand_window(self, sku: str, window: int = DEFAULT_DEMAND_WINDOW) -> tuple[dict, pd.DataFrame]:
        canonical = self._canonical_sku(sku)
        if canonical is None:
            return {"limitation": f"Unknown SKU: {sku}"}, pd.DataFrame()
        rows = self.demand_history.loc[self.demand_history["sku"] == canonical].copy()
        values = rows["demand_qty"].dropna().to_numpy(dtype=float)
        if not len(values):
            return {
                "sku": canonical,
                "history_points": 0,
                "limitation": "No demand history is available for this SKU.",
            }, pd.DataFrame()
        actual_window = min(max(int(window), 1), len(values))
        recent = values[-actual_window:]
        previous_end = len(values) - actual_window
        previous_start = max(previous_end - actual_window, 0)
        previous = values[previous_start:previous_end]
        recent_avg = float(recent.mean())
        previous_avg = float(previous.mean()) if len(previous) else np.nan
        change_pct = _safe_pct_change(recent_avg, previous_avg)
        metrics = {
            "sku": canonical,
            "period_field": "week",
            "period_kind": self.demand_period_kind,
            "history_points": int(len(values)),
            "requested_recent_periods": int(window),
            "actual_recent_periods": int(actual_window),
            "recent_average": round(recent_avg, 2),
            "previous_comparison_periods": int(len(previous)),
            "previous_average": round(previous_avg, 2) if pd.notna(previous_avg) else None,
            "recent_vs_previous_change_pct": change_pct,
            "trend": _trend_label(recent_avg, previous_avg, change_pct),
            "historical_average": round(float(values.mean()), 2),
            "historical_std": round(float(values.std(ddof=1)), 2) if len(values) > 1 else 0.0,
            "historical_cv": round(float(values.std(ddof=1) / values.mean()), 4)
            if len(values) > 1 and values.mean() != 0 else None,
        }
        if self.demand_period_kind == "source_order":
            metrics["ordering_limitation"] = (
                "The demand period field could not be parsed consistently; recent periods use uploaded row order."
            )
        table = rows.tail(actual_window)[["sku", "week", "demand_qty"]].reset_index(drop=True)
        return metrics, table

    def receipt_metrics(self, sku: str) -> dict[str, Any]:
        canonical = self._canonical_sku(sku)
        if canonical is None:
            return {"limitation": f"Unknown SKU: {sku}"}
        match = self.receipt_summary.loc[self.receipt_summary["sku"] == canonical]
        if match.empty:
            return {
                "sku": canonical,
                "receipt_count": 0,
                "limitation": "No receipt history is available for this SKU.",
            }
        result = match.iloc[0].to_dict()
        result["chronology_limitation"] = (
            "receipt_history has no order/receipt date, so recent lead-time trends cannot be established."
        )
        return result

    def sku_profile(self, sku: str, demand_window: int = DEFAULT_DEMAND_WINDOW) -> dict[str, Any] | None:
        canonical = self._canonical_sku(sku)
        if canonical is None:
            return None
        row = self.sku_view.loc[self.sku_view["sku"] == canonical].iloc[0]
        demand, _ = self.demand_window(canonical, demand_window)
        receipts = self.receipt_metrics(canonical)
        item_fields = [
            "sku", "description", "supplier", "item_class", "assumed_lead_time_days",
            "current_safety_stock", "current_rop", "unit_cost",
        ]
        analysis_fields = [
            "recommended_safety_stock", "recommended_rop", "safety_stock_gap", "rop_gap",
            "safety_stock_gap_pct", "rop_gap_pct", "service_level_z", "flagged",
            "analysis_flags", "dollar_impact", "stockout_risk_score", "stockout_risk_level",
        ]
        return {
            "item_master": {
                field: row[field] for field in item_fields if field in row.index and _present(row[field])
            },
            "demand_history": demand,
            "receipt_history": receipts,
            "analysis_results": {
                field: row[field]
                for field in analysis_fields
                if field in row.index and _present(row[field])
            },
        }

    def supplier_profile(self, supplier: str, limit: int = 15) -> dict[str, Any] | None:
        canonical = self._canonical_supplier(supplier)
        if canonical is None:
            return None
        sku_rows = self.sku_view.loc[self.sku_view["supplier"] == canonical].copy()
        skus = set(sku_rows["sku"])
        receipt_rows = self.receipt_history.loc[self.receipt_history["sku"].isin(skus)]
        lead_times = receipt_rows["actual_lead_time_days"].dropna()
        abc_counts = sku_rows["item_class"].astype(str).str.upper().value_counts().to_dict()
        a_sku_count = int(abc_counts.get("A", 0))
        under_protected_a_count = int(
            ((sku_rows["item_class"].astype(str).str.upper() == "A")
             & (sku_rows["safety_stock_gap"] > 0)).sum()
        )
        summary = {
            "supplier": canonical,
            "sku_count": int(len(sku_rows)),
            "abc_mix": {key: int(value) for key, value in abc_counts.items()},
            "receipt_count": int(len(lead_times)),
            "receipt_weighted_average_lead_time_days": round(float(lead_times.mean()), 2)
            if len(lead_times) else None,
            "receipt_weighted_median_lead_time_days": round(float(lead_times.median()), 2)
            if len(lead_times) else None,
            "receipt_level_lead_time_std_days": round(float(lead_times.std(ddof=1)), 2)
            if len(lead_times) > 1 else 0.0 if len(lead_times) else None,
            "receipt_level_lead_time_cv": round(float(lead_times.std(ddof=1) / lead_times.mean()), 4)
            if len(lead_times) > 1 and lead_times.mean() != 0 else None,
            "skus_with_increasing_demand": int((sku_rows["demand_trend"] == "increasing").sum()),
            "skus_with_positive_safety_stock_gap": int((sku_rows["safety_stock_gap"] > 0).sum()),
            "under_protected_a_skus": under_protected_a_count,
            "under_protected_a_share": round(under_protected_a_count / a_sku_count, 4)
            if a_sku_count else None,
            "total_safety_stock_gap_units": round(float(sku_rows["safety_stock_gap"].sum()), 2),
            "total_positive_safety_stock_gap_units": round(
                float(sku_rows["safety_stock_gap"].clip(lower=0).sum()), 2
            ),
            "average_stockout_risk_score_across_skus": round(
                float(sku_rows["stockout_risk_score"].mean()), 2
            ),
            "definitions": {
                "lead_time_average": "weighted across individual receipt records",
                "risk_average": "simple average of SKU-level relative Stockout Risk scores",
                "supplier_mapping": "current primary supplier from item_master",
                "under_protected_a_share": "A-class SKUs with a positive Safety Stock gap divided by all A-class SKUs for the supplier",
                "positive_safety_stock_gap": "sum of recommended-minus-current Safety Stock where the gap is positive",
            },
            "sku_table": _records(
                sku_rows.sort_values("stockout_risk_score", ascending=False)[[
                    "sku", "description", "item_class", "demand_cv", "demand_trend",
                    "lead_time_cv", "safety_stock_gap", "rop_gap", "stockout_risk_score",
                ]],
                limit,
            ),
        }
        return summary

    def supplier_summary(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for supplier in self.item_master["supplier"].drop_duplicates():
            profile = self.supplier_profile(str(supplier), limit=0)
            if profile is None:
                continue
            rows.append({key: value for key, value in profile.items() if key not in {"definitions", "sku_table"}})
        return pd.DataFrame(rows)


def get_sku_profile(model: InventoryDataModel, sku: str, demand_window: int = 8):
    return model.sku_profile(sku, demand_window)


def get_supplier_profile(model: InventoryDataModel, supplier: str, limit: int = 15):
    return model.supplier_profile(supplier, limit)

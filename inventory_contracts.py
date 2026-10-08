"""Canonical data contracts for uploaded inventory data.

The upload pipeline and the analytical assistant share these definitions so a
column cannot silently acquire a different meaning in chat than it has in the
SS/ROP calculation.  Validation is intentionally strict for material errors;
non-fatal quality observations are returned as warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class TableContract:
    """Required columns, grain, and analytical meaning for one source table."""

    name: str
    grain: str
    required_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...] = ()
    identifier_columns: tuple[str, ...] = ()


TABLE_CONTRACTS: dict[str, TableContract] = {
    "item_master": TableContract(
        name="item_master",
        grain="one row per SKU",
        required_columns=(
            "sku", "description", "supplier", "item_class",
            "assumed_lead_time_days", "current_safety_stock", "current_rop", "unit_cost",
        ),
        numeric_columns=(
            "assumed_lead_time_days", "current_safety_stock", "current_rop", "unit_cost",
        ),
        identifier_columns=("sku", "supplier"),
    ),
    "demand_history": TableContract(
        name="demand_history",
        grain="one row per SKU and week/period",
        required_columns=("sku", "week", "demand_qty"),
        numeric_columns=("demand_qty",),
        identifier_columns=("sku",),
    ),
    "receipt_history": TableContract(
        name="receipt_history",
        grain=(
            "one receipt observation; a PO may have multiple rows when it has partial receipts"
        ),
        required_columns=("sku", "po_number", "actual_lead_time_days"),
        numeric_columns=("actual_lead_time_days",),
        identifier_columns=("sku", "po_number"),
    ),
}


DATA_DICTIONARY: dict[str, dict[str, str]] = {
    "item_master": {
        "sku": "Item identifier; primary key of item_master.",
        "description": "Human-readable item description.",
        "supplier": "Current primary supplier mapping, not historical PO supplier.",
        "item_class": "Input ABC class supplied by the uploader; it is not recomputed.",
        "assumed_lead_time_days": "Current planning lead time in calendar days.",
        "current_safety_stock": "Current safety-stock quantity in the item's planning unit.",
        "current_rop": "Current reorder-point quantity in the item's planning unit.",
        "unit_cost": "Cost per planning unit in the uploader's consistent currency.",
    },
    "demand_history": {
        "sku": "Item identifier joining to item_master.sku.",
        "week": "Ordered demand period; numeric week or parseable date.",
        "demand_qty": "Non-negative demand quantity for the period.",
    },
    "receipt_history": {
        "sku": "Item identifier joining to item_master.sku.",
        "po_number": "Source PO identifier; repeated values may represent partial receipts.",
        "actual_lead_time_days": "Positive observed receipt lead time in calendar days.",
    },
    "analysis_results": {
        "recommended_safety_stock": "SS/ROP engine output using the documented combined-variability formula.",
        "recommended_rop": "Mean lead-time demand plus recommended safety stock.",
    },
}


@dataclass
class ValidationReport:
    """Quality metadata retained with a validated dataset."""

    row_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    duplicate_counts: dict[str, int] = field(default_factory=dict)


class DataContractError(ValueError):
    """Raised when source data cannot be interpreted without guessing."""


def _clean_identifier(frame: pd.DataFrame, table: str, column: str) -> None:
    if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
        raise DataContractError(f"{table} contains a missing or blank {column}.")
    frame[column] = frame[column].astype(str).str.strip()


def validate_inventory_frames(
    item_master: pd.DataFrame,
    demand_history: pd.DataFrame,
    receipt_history: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], ValidationReport]:
    """Validate and normalize the three canonical input tables.

    Repeated PO numbers are retained because the current schema describes
    receipt rows, not guaranteed one-row-per-PO orders. Exact duplicate receipt
    rows are rejected because they cannot be distinguished from accidental
    duplication and would bias lead-time statistics.
    """

    frames = {
        "item_master": item_master.copy(),
        "demand_history": demand_history.copy(),
        "receipt_history": receipt_history.copy(),
    }
    warnings: list[str] = []
    duplicate_counts: dict[str, int] = {}
    for name, frame in frames.items():
        contract = TABLE_CONTRACTS[name]
        missing = set(contract.required_columns) - set(frame.columns)
        if missing:
            raise DataContractError(
                f"{name} is missing required column(s): {', '.join(sorted(missing))}"
            )
        if frame.empty:
            raise DataContractError(f"{name} contains no data rows.")
        for column in contract.identifier_columns:
            _clean_identifier(frame, name, column)
        for column in contract.numeric_columns:
            try:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise DataContractError(f"{name}.{column} contains a non-numeric value.") from exc
            if frame[column].isna().any():
                raise DataContractError(f"{name}.{column} contains a null value.")

    items = frames["item_master"]
    demand = frames["demand_history"]
    receipts = frames["receipt_history"]
    if items["sku"].duplicated().any():
        values = sorted(items.loc[items["sku"].duplicated(False), "sku"].unique())
        raise DataContractError(
            "item_master must contain one row per SKU. Duplicate SKU(s): "
            + ", ".join(values[:10])
        )
    invalid_classes = sorted(set(items["item_class"].astype(str).str.upper()) - {"A", "B", "C"})
    if invalid_classes:
        warnings.append(
            "Unknown item_class values use the calculation engine's Class B service factor: "
            + ", ".join(invalid_classes[:10])
        )
    if (items[["assumed_lead_time_days", "unit_cost"]] <= 0).any().any():
        raise DataContractError("item_master lead time and unit cost must be positive.")
    if (items[["current_safety_stock", "current_rop"]] < 0).any().any():
        raise DataContractError("item_master stock quantities must be non-negative.")
    if demand[["sku", "week"]].duplicated().any():
        raise DataContractError("demand_history contains duplicate SKU/week rows.")
    if demand["week"].isna().any() or demand["week"].astype(str).str.strip().eq("").any():
        raise DataContractError("demand_history.week contains a missing or blank value.")
    if (demand["demand_qty"] < 0).any():
        raise DataContractError("demand_history.demand_qty must be non-negative.")
    week_text = demand["week"].astype(str).str.strip()
    numeric_week = pd.to_numeric(week_text, errors="coerce")
    if not numeric_week.notna().all():
        date_week = pd.to_datetime(week_text, errors="coerce", format="mixed")
        if not date_week.notna().all():
            raise DataContractError("demand_history.week must be consistently numeric or date-like.")
    if (receipts["actual_lead_time_days"] <= 0).any():
        raise DataContractError("receipt_history.actual_lead_time_days must be positive.")
    exact_receipt_duplicates = int(
        receipts.duplicated(["sku", "po_number", "actual_lead_time_days"]).sum()
    )
    duplicate_counts["receipt_exact_rows"] = exact_receipt_duplicates
    if exact_receipt_duplicates:
        raise DataContractError(
            "receipt_history contains exact duplicate SKU/PO/lead-time rows; resolve them before analysis."
        )
    repeated_po_rows = int(receipts.duplicated(["sku", "po_number"], keep=False).sum())
    duplicate_counts["receipt_rows_with_repeated_po"] = repeated_po_rows
    if repeated_po_rows:
        warnings.append(
            f"{repeated_po_rows} receipt rows share a SKU/PO identifier. They are retained as "
            "partial-receipt observations; PO-level lead time is not inferred."
        )

    master_skus = set(items["sku"])
    for name, frame in (("demand_history", demand), ("receipt_history", receipts)):
        unknown = sorted(set(frame["sku"]) - master_skus)
        missing_history = sorted(master_skus - set(frame["sku"]))
        if unknown:
            raise DataContractError(
                f"{name} contains SKU(s) absent from item_master: " + ", ".join(unknown[:10])
            )
        if missing_history:
            raise DataContractError(
                f"{name} has no history for item-master SKU(s): "
                + ", ".join(missing_history[:10])
            )

    receipts = receipts.reset_index(drop=True)
    receipts["receipt_row_id"] = [f"receipt-{index + 1}" for index in range(len(receipts))]
    frames["receipt_history"] = receipts
    return frames, ValidationReport(
        row_counts={name: len(frame) for name, frame in frames.items()},
        warnings=warnings,
        duplicate_counts=duplicate_counts,
    )


def dataset_fingerprint(tables: Mapping[str, pd.DataFrame]) -> str:
    """Return a stable content fingerprint without serializing raw rows to logs."""

    digest = hashlib.sha256(b"inventory-dataset-v1")
    for name in sorted(tables):
        frame = tables[name]
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(list(frame.columns), separators=(",", ":")).encode("utf-8"))
        row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
        digest.update(row_hashes)
    return digest.hexdigest()


def schema_for_prompt(tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Return the exact, non-row schema supplied to the planning model."""

    result: dict[str, Any] = {}
    for name, frame in tables.items():
        contract = TABLE_CONTRACTS.get(name)
        result[name] = {
            "grain": contract.grain if contract else "one row per SKU (derived)",
            "row_count": int(len(frame)),
            "columns": {
                column: {
                    "dtype": str(frame[column].dtype),
                    "description": DATA_DICTIONARY.get(name, {}).get(column, "Derived analytical field."),
                }
                for column in frame.columns
            },
        }
    return result

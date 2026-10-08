"""Bounded analytical execution and typed evidence for inventory questions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import threading
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from inventory_contracts import dataset_fingerprint
from inventory_data import InventoryDataModel


MAX_RESULT_ROWS = 100
MAX_DISPLAY_ROWS = 25
DEFAULT_TIMEOUT_SECONDS = 5.0
ALLOWED_TABLES = {
    "item_master", "demand_history", "receipt_history", "analysis_results",
    "sku_metrics", "demand_metrics", "receipt_metrics",
}
ALLOWED_FUNCTIONS = {
    "abs", "avg", "coalesce", "concat", "count", "date_diff", "greatest", "hash", "least",
    "lower", "max", "median", "min", "nullif", "round", "stddev_samp",
    "sum", "upper",
}


class AnalyticalExecutionError(RuntimeError):
    """Safe failure raised by the analytical boundary."""


class SQLValidationError(AnalyticalExecutionError):
    """Raised when a generated query exceeds the read-only SQL policy."""


@dataclass
class EvidenceResult:
    """Executor-authored evidence; model claims cannot populate coverage fields."""

    result_id: str
    dataset_fingerprint: str
    operation: str
    source_tables: list[str]
    metric_definition: str | None = None
    units: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    date_interval: dict[str, Any] | None = None
    eligible_entity_count: int = 0
    evaluated_entity_count: int = 0
    excluded_count: int = 0
    excluded_reasons: dict[str, int] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    source_rows: list[dict[str, Any]] = field(default_factory=list)
    tie_handling: str = "not applicable"
    calculation_complete: bool = True
    display_truncated: bool = False
    query: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


METRICS: dict[str, dict[str, Any]] = {
    "lead_time_variability": {
        "column": "lead_time_std", "minimum": 2, "units": "calendar days",
        "definition": "Sample standard deviation of receipt-row actual lead times (ddof=1).",
        "sources": ["receipt_history"],
    },
    "mean_lead_time": {
        "column": "lead_time_mean", "minimum": 1, "units": "calendar days",
        "definition": "Arithmetic mean of receipt-row actual lead times.",
        "sources": ["receipt_history"],
    },
    "longest_individual_lead_time": {
        "column": "actual_lead_time_days", "minimum": 1, "units": "calendar days",
        "definition": "Maximum individual receipt-row actual lead time; not a PO-level aggregate.",
        "sources": ["receipt_history"],
    },
    "demand_variability": {
        "column": "demand_std", "minimum": 2, "units": "demand units per period",
        "definition": "Sample standard deviation of period demand (ddof=1).",
        "sources": ["demand_history"],
    },
    "mean_demand": {
        "column": "demand_mean", "minimum": 1, "units": "demand units per period",
        "definition": "Arithmetic mean of demand quantity across uploaded periods.",
        "sources": ["demand_history"],
    },
    "safety_stock_gap": {
        "column": "safety_stock_gap", "minimum": 1, "units": "inventory units",
        "definition": "Recommended safety stock minus current safety stock.",
        "sources": ["item_master", "demand_history", "receipt_history", "analysis_results"],
    },
    "rop_gap": {
        "column": "rop_gap", "minimum": 1, "units": "inventory units",
        "definition": "Recommended reorder point minus current reorder point.",
        "sources": ["item_master", "demand_history", "receipt_history", "analysis_results"],
    },
    "stockout_risk": {
        "column": "stockout_risk_score", "minimum": 1, "units": "relative score (0-100)",
        "definition": "Documented relative planning score; it is not historical stockout evidence.",
        "sources": ["item_master", "demand_history", "receipt_history", "analysis_results"],
    },
}


def _clean_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _new_result(fingerprint: str, operation: str, **kwargs: Any) -> EvidenceResult:
    return EvidenceResult(
        result_id=f"result-{uuid.uuid4().hex[:12]}",
        dataset_fingerprint=fingerprint,
        operation=operation,
        **kwargs,
    )


def validate_read_only_sql(sql: str) -> Any:
    """Parse and structurally validate one SELECT/CTE query.

    This deliberately uses the AST rather than keyword filtering. Query text is
    never executed when parsing or allow-list checks fail.
    """

    try:
        from sqlglot import exp, parse
    except ImportError as exc:
        raise SQLValidationError("sqlglot is required for structural SQL validation.") from exc
    try:
        statements = parse(sql, read="duckdb")
    except Exception as exc:
        raise SQLValidationError(f"SQL could not be parsed: {exc}") from exc
    if len(statements) != 1:
        raise SQLValidationError("Exactly one SQL statement is allowed.")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise SQLValidationError("Only a SELECT query, optionally with CTEs, is allowed.")
    prohibited_names = (
        "Insert", "Update", "Delete", "Create", "Drop", "Alter", "Command",
        "Attach", "Detach", "Copy", "Merge", "Pragma", "Install", "LoadData",
    )
    prohibited_types = tuple(
        node_type for name in prohibited_names if (node_type := getattr(exp, name, None))
    )
    if prohibited_types and any(tree.find_all(*prohibited_types)):
        raise SQLValidationError("The query contains a prohibited SQL operation.")
    if any(tree.find_all(exp.Limit)):
        raise SQLValidationError(
            "SQL LIMIT is not allowed; the executor applies a display limit after full calculation."
        )
    cte_names = {cte.alias_or_name.casefold() for cte in tree.find_all(exp.CTE)}
    referenced = {table.name.casefold() for table in tree.find_all(exp.Table)}
    invalid_tables = referenced - ALLOWED_TABLES - cte_names
    if invalid_tables:
        raise SQLValidationError(
            "Query references unapproved table(s): " + ", ".join(sorted(invalid_tables))
        )
    for function in tree.find_all(exp.Func):
        name = function.sql_name().casefold()
        if name == "anonymous":
            name = str(getattr(function, "name", "")).casefold()
        if name not in ALLOWED_FUNCTIONS:
            raise SQLValidationError(f"SQL function is not approved: {name}")
    return tree


class InventoryAnalytics:
    """Dataset-scoped deterministic analytical interface."""

    def __init__(self, model: InventoryDataModel, validation_warnings: Sequence[str] = ()):
        self.model = model
        self.tables = {
            "item_master": model.item_master,
            "demand_history": model.demand_history.drop(
                columns=["_source_order", "_period_sort"], errors="ignore"
            ),
            "receipt_history": model.receipt_history,
            "analysis_results": model.analysis_results,
            "sku_metrics": model.sku_view,
            "demand_metrics": model.demand_summary,
            "receipt_metrics": model.receipt_summary,
        }
        self.fingerprint = dataset_fingerprint(
            {name: self.tables[name] for name in ("item_master", "demand_history", "receipt_history")}
        )
        self.validation_warnings = list(validation_warnings)

    def _filtered_sku_view(self, filters: Mapping[str, Any] | None) -> pd.DataFrame:
        frame = self.model.sku_view.copy()
        for key, value in dict(filters or {}).items():
            if key not in {"sku", "supplier", "item_class"}:
                raise AnalyticalExecutionError(f"Unsupported filter: {key}")
            values = value if isinstance(value, list) else [value]
            wanted = {str(item).strip().casefold() for item in values}
            if key not in frame.columns:
                raise AnalyticalExecutionError(f"Filter field is unavailable: {key}")
            frame = frame.loc[frame[key].astype(str).str.casefold().isin(wanted)]
        return frame

    def rank_skus(
        self,
        metric: str,
        *,
        direction: str = "descending",
        limit: int = 1,
        filters: Mapping[str, Any] | None = None,
    ) -> EvidenceResult:
        """Rank all eligible SKUs before applying a display limit, retaining ties."""

        if metric not in METRICS or metric == "longest_individual_lead_time":
            raise AnalyticalExecutionError(f"Unsupported SKU ranking metric: {metric}")
        if direction not in {"ascending", "descending"}:
            raise AnalyticalExecutionError("direction must be ascending or descending.")
        if not 1 <= int(limit) <= MAX_DISPLAY_ROWS:
            raise AnalyticalExecutionError(f"limit must be between 1 and {MAX_DISPLAY_ROWS}.")
        spec = METRICS[metric]
        population = self._filtered_sku_view(filters)
        eligible = population.copy()
        exclusion_reasons: dict[str, int] = {}
        if metric.startswith("lead_time") or metric == "mean_lead_time":
            count_column = "receipt_count"
        elif metric.startswith("demand") or metric == "mean_demand":
            count_column = "demand_history_points"
        else:
            count_column = None
        if count_column:
            insufficient = pd.to_numeric(eligible[count_column], errors="coerce").fillna(0) < spec["minimum"]
            exclusion_reasons[f"fewer than {spec['minimum']} observations"] = int(insufficient.sum())
            eligible = eligible.loc[~insufficient]
        missing = pd.to_numeric(eligible[spec["column"]], errors="coerce").isna()
        if missing.any():
            exclusion_reasons["metric unavailable"] = int(missing.sum())
            eligible = eligible.loc[~missing]
        ordered = eligible.sort_values(
            [spec["column"], "sku"], ascending=[direction == "ascending", True], kind="stable"
        )
        requested = int(limit)
        selected = ordered.head(requested)
        tie_note = "All rows tied at the cutoff are returned."
        if not selected.empty and len(ordered) > requested:
            cutoff = selected.iloc[-1][spec["column"]]
            ties = ordered.loc[ordered[spec["column"]] == cutoff]
            selected = pd.concat([selected, ties]).drop_duplicates(subset=["sku"])
        columns = [
            "sku", "description", "supplier", "item_class", count_column,
            spec["column"],
        ]
        columns = [column for column in columns if column and column in selected.columns]
        return _new_result(
            self.fingerprint,
            "rank_skus",
            source_tables=spec["sources"],
            metric_definition=spec["definition"],
            units=spec["units"],
            filters=dict(filters or {}),
            eligible_entity_count=int(len(eligible)),
            evaluated_entity_count=int(len(eligible)),
            excluded_count=int(len(population) - len(eligible)),
            excluded_reasons={k: v for k, v in exclusion_reasons.items() if v},
            rows=_clean_records(selected[columns]),
            tie_handling=tie_note,
            parameters={"metric": metric, "direction": direction, "limit": requested},
            warnings=self.validation_warnings.copy(),
        )

    def longest_receipt(self, skus: Sequence[str] | None = None) -> EvidenceResult:
        """Return maximum source receipt row(s), globally or for resolved SKUs."""

        wanted = {str(sku).strip().casefold() for sku in (skus or []) if str(sku).strip()}
        available = {
            str(sku).casefold(): str(sku) for sku in self.model.item_master["sku"].astype(str)
        }
        unknown = sorted(wanted - set(available))
        if unknown:
            raise AnalyticalExecutionError("Unknown SKU(s): " + ", ".join(unknown))
        rows = self.model.receipt_history.copy()
        if wanted:
            rows = rows.loc[rows["sku"].astype(str).str.casefold().isin(wanted)]
        if rows.empty:
            return _new_result(
                self.fingerprint, "longest_receipt", source_tables=["receipt_history"],
                metric_definition=METRICS["longest_individual_lead_time"]["definition"],
                units="calendar days",
                filters={"sku": sorted(available[value] for value in wanted)} if wanted else {},
                excluded_count=len(wanted), excluded_reasons={"no receipt rows": len(wanted)},
                warnings=self.validation_warnings + ["No receipt history matched the requested SKU(s)."],
            )
        maximum = float(rows["actual_lead_time_days"].max())
        winners = rows.loc[rows["actual_lead_time_days"] == maximum].copy()
        columns = [
            column for column in ("receipt_row_id", "sku", "po_number", "actual_lead_time_days")
            if column in winners.columns
        ]
        warnings = self.validation_warnings.copy()
        if winners["po_number"].duplicated().any():
            warnings.append(
                "The winning PO has multiple tied receipt rows; the evidence identifies receipt rows, "
                "not an inferred PO-level lead time."
            )
        return _new_result(
            self.fingerprint,
            "longest_receipt",
            source_tables=["receipt_history"],
            metric_definition=METRICS["longest_individual_lead_time"]["definition"],
            units="calendar days",
            filters={"sku": sorted(available[value] for value in wanted)} if wanted else {},
            eligible_entity_count=int(rows["sku"].nunique()),
            evaluated_entity_count=int(rows["sku"].nunique()),
            rows=_clean_records(winners[columns]),
            source_rows=_clean_records(winners[columns]),
            tie_handling="All receipt rows tied for the maximum are returned.",
            parameters={"metric": "longest_individual_lead_time"},
            warnings=warnings,
        )

    def compare_skus(self, skus: Sequence[str], metrics: Sequence[str]) -> EvidenceResult:
        """Compare an exact requested SKU set without reducing coverage to one row."""

        requested = [str(sku).strip() for sku in skus if str(sku).strip()]
        if len(requested) < 2:
            raise AnalyticalExecutionError("At least two SKUs are required for comparison.")
        unknown_metrics = set(metrics) - set(METRICS)
        if unknown_metrics:
            raise AnalyticalExecutionError("Unsupported metric(s): " + ", ".join(sorted(unknown_metrics)))
        wanted = {sku.casefold() for sku in requested}
        rows = self.model.sku_view.loc[
            self.model.sku_view["sku"].astype(str).str.casefold().isin(wanted)
        ].copy()
        found = {str(sku).casefold() for sku in rows["sku"]}
        missing = sorted(wanted - found)
        columns = ["sku", "description", "supplier", "item_class"]
        for metric in metrics:
            column = METRICS[metric]["column"]
            if column in rows.columns:
                columns.append(column)
        warnings = self.validation_warnings.copy()
        if missing:
            warnings.append("Unknown requested SKU(s): " + ", ".join(missing))
        return _new_result(
            self.fingerprint,
            "compare_skus",
            source_tables=sorted({source for metric in metrics for source in METRICS[metric]["sources"]}),
            metric_definition="; ".join(METRICS[metric]["definition"] for metric in metrics),
            filters={"sku": requested},
            eligible_entity_count=len(rows),
            evaluated_entity_count=len(rows),
            excluded_count=len(missing),
            excluded_reasons={"unknown SKU": len(missing)} if missing else {},
            rows=_clean_records(rows[columns].sort_values("sku")),
            parameters={"metrics": list(metrics)},
            warnings=warnings,
        )

    def demand_growth(self, skus: Sequence[str], window: int = 6) -> EvidenceResult:
        """Compare the last N demand periods with the immediately preceding N."""

        if not 1 <= int(window) <= 26:
            raise AnalyticalExecutionError("Demand growth window must be between 1 and 26 periods.")
        requested = [str(sku).strip() for sku in skus if str(sku).strip()]
        if not requested:
            raise AnalyticalExecutionError("At least one SKU is required for demand growth.")
        records: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        excluded: dict[str, int] = {}
        interval: dict[str, Any] | None = None
        for sku in requested:
            canonical = self.model._canonical_sku(sku)
            if canonical is None:
                excluded["unknown SKU"] = excluded.get("unknown SKU", 0) + 1
                continue
            rows = self.model.demand_history.loc[self.model.demand_history["sku"] == canonical]
            if len(rows) < 2 * int(window):
                excluded["fewer than two complete windows"] = (
                    excluded.get("fewer than two complete windows", 0) + 1
                )
                continue
            recent = rows.tail(int(window))
            previous = rows.iloc[-2 * int(window):-int(window)]
            recent_mean = float(recent["demand_qty"].mean())
            previous_mean = float(previous["demand_qty"].mean())
            growth = None if previous_mean == 0 else (recent_mean - previous_mean) / previous_mean * 100
            records.append({
                "sku": canonical,
                "window_periods": int(window),
                "recent_mean_demand": recent_mean,
                "previous_mean_demand": previous_mean,
                "growth_pct": growth,
                "is_increasing": growth is not None and growth > 0,
            })
            selected = pd.concat([previous, recent])
            source_rows.extend(_clean_records(selected[["sku", "week", "demand_qty"]]))
            interval = {
                "reference": "latest uploaded demand period",
                "recent_start": recent.iloc[0]["week"],
                "recent_end": recent.iloc[-1]["week"],
                "previous_start": previous.iloc[0]["week"],
                "previous_end": previous.iloc[-1]["week"],
                "period_kind": self.model.demand_period_kind,
            }
        return _new_result(
            self.fingerprint,
            "demand_growth",
            source_tables=["demand_history"],
            metric_definition=(
                "Percentage change in mean demand for the most recent N periods versus the "
                "immediately preceding N periods. Increasing means growth_pct > 0."
            ),
            units="percent",
            filters={"sku": requested},
            date_interval=interval,
            eligible_entity_count=len(records),
            evaluated_entity_count=len(records),
            excluded_count=sum(excluded.values()),
            excluded_reasons=excluded,
            rows=records,
            source_rows=source_rows[:MAX_RESULT_ROWS],
            display_truncated=len(source_rows) > MAX_RESULT_ROWS,
            parameters={"window": int(window)},
            warnings=self.validation_warnings.copy(),
        )

    def execute_sql(
        self,
        sql: str,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        display_limit: int = MAX_DISPLAY_ROWS,
    ) -> EvidenceResult:
        """Execute one structurally approved query with a real interrupt deadline."""

        tree = validate_read_only_sql(sql)
        try:
            import duckdb
        except ImportError as exc:
            raise AnalyticalExecutionError("duckdb is required for analytical SQL execution.") from exc
        connection = duckdb.connect(database=":memory:", config={"enable_external_access": "false"})
        connection.execute("SET memory_limit='256MB'")
        connection.execute("SET threads=1")
        for name, frame in self.tables.items():
            connection.register(name, frame)
        outcome: dict[str, Any] = {}

        def run() -> None:
            try:
                outcome["frame"] = connection.execute(sql).fetchdf()
            except Exception as exc:  # surfaced below with a bounded user-safe message
                outcome["error"] = exc

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(max(float(timeout_seconds), 0.1))
        if worker.is_alive():
            connection.interrupt()
            worker.join(1.0)
            connection.close()
            raise AnalyticalExecutionError(
                f"The analytical query exceeded the {timeout_seconds:g}-second execution limit."
            )
        if "error" in outcome:
            connection.close()
            raise AnalyticalExecutionError(f"The analytical query failed: {outcome['error']}")
        frame: pd.DataFrame = outcome["frame"]
        connection.close()
        if len(frame) > MAX_RESULT_ROWS:
            raise AnalyticalExecutionError(
                f"The query returned {len(frame)} rows; the analytical maximum is {MAX_RESULT_ROWS}."
            )
        shown = frame.head(min(max(int(display_limit), 1), MAX_DISPLAY_ROWS))
        from sqlglot import exp
        referenced = sorted({table.name.casefold() for table in tree.find_all(exp.Table)})
        return _new_result(
            self.fingerprint,
            "analytical_sql",
            source_tables=[name for name in referenced if name in ALLOWED_TABLES],
            eligible_entity_count=len(frame),
            evaluated_entity_count=len(frame),
            rows=_clean_records(shown),
            calculation_complete=True,
            display_truncated=len(frame) > len(shown),
            query=sql,
            warnings=self.validation_warnings.copy(),
        )

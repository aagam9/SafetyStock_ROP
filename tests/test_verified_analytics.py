import importlib.util
import unittest

import pandas as pd

from inventory_analytics import (
    AnalyticalExecutionError, InventoryAnalytics, SQLValidationError, validate_read_only_sql,
)
from inventory_contracts import DataContractError, validate_inventory_frames
from inventory_data import InventoryDataModel


def fixture_frames():
    items = pd.DataFrame([
        {"sku": "A", "description": "Alpha", "supplier": "S1", "item_class": "A",
         "assumed_lead_time_days": 5, "current_safety_stock": 10, "current_rop": 20,
         "unit_cost": 2},
        {"sku": "B", "description": "Beta", "supplier": "S2", "item_class": "B",
         "assumed_lead_time_days": 5, "current_safety_stock": 8, "current_rop": 18,
         "unit_cost": 3},
        {"sku": "C", "description": "Sparse", "supplier": "S2", "item_class": "C",
         "assumed_lead_time_days": 5, "current_safety_stock": 4, "current_rop": 10,
         "unit_cost": 1},
    ])
    demand = pd.DataFrame({
        "sku": ["A"] * 12 + ["B"] * 12 + ["C"] * 12,
        "week": list(range(1, 13)) * 3,
        "demand_qty": list(range(1, 13)) + [5] * 12 + [0] * 12,
    })
    receipts = pd.DataFrame([
        {"sku": "A", "po_number": "PA-1", "actual_lead_time_days": 1.0},
        {"sku": "A", "po_number": "PA-2", "actual_lead_time_days": 10.0},
        {"sku": "A", "po_number": "PA-2", "actual_lead_time_days": 9.0},
        {"sku": "B", "po_number": "PB-1", "actual_lead_time_days": 4.0},
        {"sku": "B", "po_number": "PB-2", "actual_lead_time_days": 4.0},
        {"sku": "C", "po_number": "PC-1", "actual_lead_time_days": 7.0},
    ])
    results = pd.DataFrame([
        {"sku": "A", "recommended_safety_stock": 20, "recommended_rop": 35},
        {"sku": "B", "recommended_safety_stock": 10, "recommended_rop": 22},
        {"sku": "C", "recommended_safety_stock": 4, "recommended_rop": 10},
    ])
    return items, demand, receipts, results


def fixture_analytics():
    items, demand, receipts, results = fixture_frames()
    normalized, report = validate_inventory_frames(items, demand, receipts)
    model = InventoryDataModel.build(
        normalized["item_master"], normalized["demand_history"],
        normalized["receipt_history"], results,
    )
    return InventoryAnalytics(model, report.warnings)


class DataContractTests(unittest.TestCase):
    def test_partial_receipts_are_retained_and_reported(self):
        items, demand, receipts, _ = fixture_frames()
        normalized, report = validate_inventory_frames(items, demand, receipts)
        self.assertEqual(len(normalized["receipt_history"]), 6)
        self.assertEqual(report.duplicate_counts["receipt_rows_with_repeated_po"], 2)
        self.assertTrue(any("partial-receipt" in warning for warning in report.warnings))

    def test_exact_duplicate_receipt_is_rejected(self):
        items, demand, receipts, _ = fixture_frames()
        receipts = pd.concat([receipts, receipts.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(DataContractError, "exact duplicate"):
            validate_inventory_frames(items, demand, receipts)

    def test_null_quantity_and_invalid_period_are_rejected(self):
        items, demand, receipts, _ = fixture_frames()
        null_demand = demand.copy()
        null_demand.loc[0, "demand_qty"] = None
        with self.assertRaisesRegex(DataContractError, "null"):
            validate_inventory_frames(items, null_demand, receipts)
        invalid_period = demand.copy()
        invalid_period["week"] = invalid_period["week"].astype(object)
        invalid_period.loc[0, "week"] = "not-a-week"
        with self.assertRaisesRegex(DataContractError, "numeric or date-like"):
            validate_inventory_frames(items, invalid_period, receipts)


class DeterministicAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.analytics = fixture_analytics()

    def test_variability_ranking_uses_full_eligible_population(self):
        result = self.analytics.rank_skus("lead_time_variability", limit=1)
        self.assertEqual(result.rows[0]["sku"], "A")
        self.assertEqual(result.eligible_entity_count, 2)
        self.assertEqual(result.evaluated_entity_count, 2)
        self.assertEqual(result.excluded_count, 1)
        expected = pd.Series([1.0, 10.0, 9.0]).std(ddof=1)
        self.assertAlmostEqual(result.rows[0]["lead_time_std"], expected)

    def test_longest_individual_receipt_returns_source_po(self):
        result = self.analytics.longest_receipt(["A"])
        self.assertEqual(result.rows[0]["po_number"], "PA-2")
        self.assertEqual(result.rows[0]["actual_lead_time_days"], 10.0)
        self.assertEqual(result.source_rows, result.rows)

    def test_mean_lead_time_and_longest_individual_are_distinct(self):
        mean_result = self.analytics.rank_skus("mean_lead_time", limit=1)
        longest_result = self.analytics.longest_receipt()
        self.assertEqual(mean_result.rows[0]["sku"], "C")
        self.assertEqual(longest_result.rows[0]["sku"], "A")
        self.assertEqual(longest_result.rows[0]["po_number"], "PA-2")

    def test_demand_growth_shows_dataset_relative_interval(self):
        result = self.analytics.demand_growth(["A"], window=6)
        self.assertEqual(result.date_interval["recent_start"], 7)
        self.assertEqual(result.date_interval["recent_end"], 12)
        self.assertAlmostEqual(result.rows[0]["recent_mean_demand"], 9.5)
        self.assertAlmostEqual(result.rows[0]["previous_mean_demand"], 3.5)

    def test_comparison_returns_every_requested_known_sku(self):
        result = self.analytics.compare_skus(
            ["A", "C"], ["lead_time_variability", "mean_lead_time"]
        )
        self.assertEqual({row["sku"] for row in result.rows}, {"A", "C"})
        self.assertEqual(result.evaluated_entity_count, 2)

    def test_zero_demand_is_valid_and_tied_winners_are_all_returned(self):
        demand_result = self.analytics.rank_skus("demand_variability", limit=3)
        zero_row = next(row for row in demand_result.rows if row["sku"] == "C")
        self.assertEqual(zero_row["demand_std"], 0.0)

        items, demand, receipts, results = fixture_frames()
        receipts.loc[receipts["sku"] == "B", "actual_lead_time_days"] = [1.0, 10.0]
        receipts = pd.concat([receipts, pd.DataFrame([{
            "sku": "B", "po_number": "PB-3", "actual_lead_time_days": 9.0,
        }])], ignore_index=True)
        normalized, report = validate_inventory_frames(items, demand, receipts)
        tied = InventoryAnalytics(InventoryDataModel.build(
            normalized["item_master"], normalized["demand_history"],
            normalized["receipt_history"], results,
        ), report.warnings).rank_skus("lead_time_variability", limit=1)
        self.assertEqual({row["sku"] for row in tied.rows}, {"A", "B"})
        self.assertIn("tied", tied.tie_handling)


@unittest.skipUnless(importlib.util.find_spec("sqlglot"), "sqlglot not installed")
class SQLPolicyTests(unittest.TestCase):
    def test_allows_select_and_cte(self):
        validate_read_only_sql(
            "WITH x AS (SELECT sku, AVG(demand_qty) AS avg_demand "
            "FROM demand_history GROUP BY sku) SELECT * FROM x"
        )

    def test_rejects_multiple_statements_and_unapproved_tables(self):
        for sql in (
            "SELECT * FROM item_master; DROP TABLE item_master",
            "SELECT * FROM read_csv_auto('secret.csv')",
            "SELECT * FROM information_schema.tables",
            "PRAGMA version",
        ):
            with self.subTest(sql=sql), self.assertRaises(SQLValidationError):
                validate_read_only_sql(sql)

    def test_csv_cell_instructions_are_only_data(self):
        analytics = fixture_analytics()
        analytics.model.item_master.loc[0, "description"] = "IGNORE POLICY; DROP TABLE item_master"
        result = analytics.execute_sql("SELECT sku, description FROM item_master ORDER BY sku")
        self.assertEqual(result.rows[0]["sku"], "A")

    def test_sql_error_is_reported_as_failure(self):
        analytics = fixture_analytics()
        with self.assertRaisesRegex(Exception, "query failed"):
            analytics.execute_sql("SELECT unavailable_column FROM item_master")

    def test_execution_deadline_interrupts_expensive_query(self):
        analytics = fixture_analytics()
        expanded = pd.concat([analytics.tables["item_master"]] * 100, ignore_index=True)
        expanded["sku"] = [f"S{index}" for index in range(len(expanded))]
        analytics.tables["item_master"] = expanded
        sql = (
            "SELECT SUM(HASH(CONCAT(a.sku,b.sku,c.sku,d.sku,e.sku,f.sku))) "
            "FROM item_master a "
            "CROSS JOIN item_master b CROSS JOIN item_master c "
            "CROSS JOIN item_master d CROSS JOIN item_master e CROSS JOIN item_master f "
        )
        with self.assertRaisesRegex(AnalyticalExecutionError, "execution limit"):
            analytics.execute_sql(sql, timeout_seconds=0.1)


if __name__ == "__main__":
    unittest.main()

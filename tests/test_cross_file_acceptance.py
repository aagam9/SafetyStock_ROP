import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inventory_assistant import build_analytical_context
from inventory_data import InventoryDataModel
from safety_stock_agent import BASE_DIR, run_analysis


class CrossFileAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp = tempfile.TemporaryDirectory()
        output = Path(cls._temp.name) / "acceptance.xlsx"
        with patch("safety_stock_agent.API_KEY", None):
            cls.result = run_analysis(
                BASE_DIR,
                api_key="",
                output_path=output,
                progress_callback=lambda _message: None,
            )
        cls.model = InventoryDataModel.build(
            cls.result["item_master"],
            cls.result["demand_history"],
            cls.result["receipt_history"],
            cls.result["analysis_results"],
        )

    @classmethod
    def tearDownClass(cls):
        cls._temp.cleanup()

    def context(self, question):
        return build_analytical_context(
            question,
            self.result["item_master"],
            self.result["demand_history"],
            self.result["receipt_history"],
            self.result["analysis_results"],
            data_model=self.model,
        )

    def test_item_master_only_supplier_lookup(self):
        _, table, metadata, _ = self.context("What supplier provides SKU MAT-1001?")
        self.assertEqual(metadata["sources_used"], ["item_master"])
        self.assertEqual(table.iloc[0]["supplier"], "Apex Components")

    def test_demand_only_last_eight_periods(self):
        _, table, metadata, _ = self.context(
            "What has demand for SKU MAT-1001 looked like during the last 8 weeks?"
        )
        self.assertEqual(metadata["sources_used"], ["demand_history"])
        self.assertEqual(len(table), 8)
        self.assertIn("demand_qty", table.columns)

    def test_receipt_only_lead_time_metrics(self):
        context, table, metadata, _ = self.context(
            "What is the average and variability of actual lead time for SKU MAT-1001?"
        )
        self.assertEqual(metadata["sources_used"], ["receipt_history"])
        self.assertIn("lead_time_mean", table.columns)
        self.assertIn("lead_time_std", table.columns)
        self.assertNotIn("recommended_safety_stock", context)

    def test_safety_stock_explanation_uses_all_sources(self):
        context, table, metadata, _ = self.context(
            "Why is the recommended Safety Stock for SKU MAT-1001 different from current?"
        )
        self.assertEqual(
            metadata["sources_used"],
            ["item_master", "demand_history", "receipt_history", "analysis_results"],
        )
        for column in ("demand_cv", "lead_time_cv", "safety_stock_gap"):
            self.assertIn(column, table.columns)
        self.assertIn("analysis_results", context)

    def test_cross_file_a_class_rising_demand_and_variable_lead_time(self):
        _, table, metadata, _ = self.context(
            "Which A-class SKUs have increasing demand and high lead-time variability?"
        )
        self.assertEqual(metadata["intent"], "cross_file_risk")
        self.assertTrue((table["item_class"] == "A").all())
        self.assertTrue((table["demand_trend"] == "increasing").all())
        self.assertIn("lead_time_cv", table.columns)

    def test_supplier_safety_stock_gap_ranking(self):
        _, table, metadata, _ = self.context(
            "Which Apex Components SKUs have the largest Safety Stock gaps?"
        )
        self.assertEqual(metadata["intent"], "supplier_inventory")
        self.assertTrue((table["supplier"] == "Apex Components").all())
        self.assertTrue(table["safety_stock_gap"].is_monotonic_decreasing)

    def test_supplier_cross_domain_filter(self):
        _, table, metadata, _ = self.context(
            "Which Apex Components SKUs have both volatile demand and inconsistent receipts?"
        )
        self.assertEqual(metadata["intent"], "cross_file_risk")
        self.assertTrue((table["supplier"] == "Apex Components").all())
        self.assertIn("demand_cv", table.columns)
        self.assertIn("lead_time_cv", table.columns)

    def test_supplier_comparison(self):
        _, table, metadata, _ = self.context(
            "Compare Apex Components and BlueRiver Industrial on lead-time consistency and inventory risk."
        )
        self.assertEqual(metadata["intent"], "supplier_comparison")
        self.assertEqual(set(table["supplier"]), {"Apex Components", "BlueRiver Industrial"})
        self.assertIn("receipt_level_lead_time_cv", table.columns)

    def test_stable_demand_with_large_safety_stock_increases(self):
        _, table, metadata, _ = self.context(
            "Which SKUs have stable historical demand but large Safety Stock increases?"
        )
        self.assertEqual(metadata["intent"], "cross_file_risk")
        self.assertIn("demand_cv", table.columns)
        self.assertTrue(table["safety_stock_gap"].is_monotonic_decreasing)

    def test_dual_demand_and_supply_risk(self):
        _, table, metadata, _ = self.context(
            "Which products show risk from both the demand side and the supply side?"
        )
        self.assertEqual(metadata["intent"], "cross_file_risk")
        self.assertIn("demand_cv", table.columns)
        self.assertIn("lead_time_cv", table.columns)

    def test_supplier_level_safety_stock_increase_ranking(self):
        _, table, metadata, _ = self.context(
            "Which suppliers are associated with the highest Safety Stock increases?"
        )
        self.assertEqual(metadata["intent"], "supplier_overview")
        self.assertIn("total_positive_safety_stock_gap_units", table.columns)
        self.assertTrue(table["total_positive_safety_stock_gap_units"].is_monotonic_decreasing)

    def test_supplier_under_protected_a_class_concentration(self):
        _, table, metadata, _ = self.context(
            "Which supplier has the largest concentration of under-protected A-class SKUs?"
        )
        self.assertEqual(metadata["intent"], "supplier_overview")
        self.assertIn("under_protected_a_share", table.columns)
        self.assertTrue(table["under_protected_a_share"].is_monotonic_decreasing)

    def test_normal_language_lead_time_variability_ranking(self):
        _, table, metadata, direct = self.context(
            "Which SKU is having the most Lead time variability?"
        )
        expected = self.model.receipt_summary.sort_values(
            "lead_time_std", ascending=False
        ).iloc[0]
        self.assertIsNone(direct)
        self.assertEqual(metadata["intent"], "receipt_ranking")
        self.assertEqual(metadata["evidence"]["question_plan"]["query_type"], "ranking")
        self.assertEqual(metadata["evidence"]["question_plan"]["metric"], "lead_time_variability")
        self.assertEqual(metadata["sources_used"], ["receipt_history"])
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]["sku"], expected["sku"])
        self.assertEqual(table.iloc[0]["lead_time_std"], expected["lead_time_std"])

    def test_ranked_sku_follow_up_pronouns_requery_data(self):
        _, first_table, first_metadata, _ = self.context(
            "Which SKU has the highest lead-time variability?"
        )
        focus_sku = first_table.iloc[0]["sku"]
        _, supplier_table, supplier_metadata, _ = build_analytical_context(
            "What supplier is it from?",
            self.result["item_master"], self.result["demand_history"],
            self.result["receipt_history"], self.result["analysis_results"],
            prior_metadata=first_metadata, data_model=self.model,
        )
        self.assertEqual(supplier_metadata["intent"], "supplier_lookup")
        self.assertEqual(supplier_table.iloc[0]["sku"], focus_sku)

        _, demand_table, demand_metadata, _ = build_analytical_context(
            "What about its demand?",
            self.result["item_master"], self.result["demand_history"],
            self.result["receipt_history"], self.result["analysis_results"],
            prior_metadata=supplier_metadata, data_model=self.model,
        )
        self.assertEqual(demand_metadata["intent"], "sku_demand")
        self.assertTrue((demand_table["sku"] == focus_sku).all())

        _, risk_table, risk_metadata, _ = build_analytical_context(
            "Why is it risky?",
            self.result["item_master"], self.result["demand_history"],
            self.result["receipt_history"], self.result["analysis_results"],
            prior_metadata=demand_metadata, data_model=self.model,
        )
        self.assertEqual(risk_metadata["intent"], "stockout_risk")
        self.assertEqual(risk_table.iloc[0]["sku"], focus_sku)


if __name__ == "__main__":
    unittest.main()

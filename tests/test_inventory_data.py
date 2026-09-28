import unittest

import pandas as pd

from inventory_data import InventoryDataError, InventoryDataModel


class InventoryRelationshipTests(unittest.TestCase):
    def setUp(self):
        self.items = pd.DataFrame([
            {"sku": "A", "description": "Alpha", "supplier": "Supplier One",
             "item_class": "A", "current_safety_stock": 20, "current_rop": 70},
            {"sku": "B", "description": "Beta", "supplier": "Supplier One",
             "item_class": "B", "current_safety_stock": 40, "current_rop": 90},
        ])
        self.demand = pd.DataFrame({
            "sku": ["A", "A", "A", "A", "B", "B"],
            "week": [1, 2, 3, 4, 1, 2],
            "demand_qty": [10, 20, 30, 40, 5, 5],
        })
        self.receipts = pd.DataFrame({
            "sku": ["A", "A", "A", "B"],
            "po_number": ["P1", "P2", "P3", "P4"],
            "actual_lead_time_days": [8, 10, 12, 6],
        })
        self.results = [
            {"sku": "A", "recommended_safety_stock": 50, "recommended_rop": 110,
             "recent_weekly_mean": 999, "analysis_flags": ["lead-time drift"]},
            {"sku": "B", "recommended_safety_stock": 30, "recommended_rop": 80,
             "recent_weekly_mean": 999, "analysis_flags": []},
        ]

    def test_histories_are_aggregated_before_relationship_join(self):
        model = InventoryDataModel.build(self.items, self.demand, self.receipts, self.results)
        self.assertEqual(len(model.sku_view), 2)
        sku_a = model.sku_view.loc[model.sku_view["sku"] == "A"].iloc[0]
        self.assertEqual(sku_a["demand_history_points"], 4)
        self.assertEqual(sku_a["receipt_count"], 3)
        self.assertEqual(sku_a["demand_total"], 100)
        self.assertEqual(sku_a["lead_time_mean"], 10)

    def test_source_data_is_authoritative_and_analysis_enriches_it(self):
        model = InventoryDataModel.build(self.items, self.demand, self.receipts, self.results)
        sku_a = model.sku_view.loc[model.sku_view["sku"] == "A"].iloc[0]
        self.assertEqual(sku_a["demand_mean"], 25)
        self.assertNotEqual(sku_a["demand_mean"], 999)
        self.assertEqual(sku_a["recommended_safety_stock"], 50)
        self.assertEqual(sku_a["safety_stock_gap"], 30)

    def test_supplier_mapping_reaches_both_fact_aggregates(self):
        model = InventoryDataModel.build(self.items, self.demand, self.receipts, self.results)
        profile = model.supplier_profile("supplier one")
        self.assertEqual(profile["sku_count"], 2)
        self.assertEqual(profile["receipt_count"], 4)
        self.assertEqual(profile["abc_mix"], {"A": 1, "B": 1})

    def test_duplicate_item_master_sku_is_rejected(self):
        duplicate = pd.concat([self.items, self.items.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(InventoryDataError, "one row per SKU"):
            InventoryDataModel.build(duplicate, self.demand, self.receipts, self.results)

    def test_missing_history_does_not_fabricate_metrics(self):
        extra = pd.DataFrame([{
            "sku": "C", "description": "Gamma", "supplier": "Supplier Two",
            "item_class": "C", "current_safety_stock": 5, "current_rop": 10,
        }])
        items = pd.concat([self.items, extra], ignore_index=True)
        results = self.results + [{"sku": "C", "recommended_safety_stock": 7, "recommended_rop": 12}]
        model = InventoryDataModel.build(items, self.demand, self.receipts, results)
        profile = model.sku_profile("C")
        self.assertEqual(profile["demand_history"]["history_points"], 0)
        self.assertIn("No demand history", profile["demand_history"]["limitation"])
        self.assertEqual(profile["receipt_history"]["receipt_count"], 0)
        self.assertIn("No receipt history", profile["receipt_history"]["limitation"])


if __name__ == "__main__":
    unittest.main()

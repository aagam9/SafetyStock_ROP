import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from safety_stock_agent import BASE_DIR, SafetyStockInputError, _load_input_data, run_analysis


class SafetyStockValidationTests(unittest.TestCase):
    def _write_inputs(self, folder: Path, supplier="Supplier One"):
        pd.DataFrame(
            [{
                "sku": "A", "description": "Alpha", "supplier": supplier,
                "item_class": "A", "assumed_lead_time_days": 10,
                "current_safety_stock": 5, "current_rop": 20, "unit_cost": 2,
            }]
        ).to_csv(folder / "item_master.csv", index=False)
        pd.DataFrame(
            {"sku": ["A", "A", "A"], "week": [1, 2, 3], "demand_qty": [10, 12, 14]}
        ).to_csv(folder / "demand_history.csv", index=False)
        pd.DataFrame(
            {"sku": ["A", "A"], "po_number": ["P1", "P2"],
             "actual_lead_time_days": [9, 11]}
        ).to_csv(folder / "receipt_history.csv", index=False)

    def test_supplier_is_required_and_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._write_inputs(folder)
            items, _, _ = _load_input_data(folder)
            self.assertEqual(items.iloc[0]["supplier"], "Supplier One")

            items.drop(columns=["supplier"]).to_csv(folder / "item_master.csv", index=False)
            with self.assertRaisesRegex(SafetyStockInputError, "supplier"):
                _load_input_data(folder)

    def test_blank_supplier_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._write_inputs(folder, supplier=" ")
            with self.assertRaisesRegex(SafetyStockInputError, "blank supplier"):
                _load_input_data(folder)

    def test_supplier_flows_through_analysis_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._write_inputs(folder)
            result = run_analysis(folder, api_key="", output_path=folder / "report.xlsx")
            self.assertEqual(result["all_results"][0]["supplier"], "Supplier One")
            self.assertTrue((folder / "report.xlsx").exists())
            portfolio = pd.read_excel(folder / "report.xlsx", sheet_name="Portfolio Scan")
            self.assertIn("Supplier", portfolio.columns)

    def test_bundled_sample_runs_end_to_end_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmp, patch("safety_stock_agent.API_KEY", None):
            output = Path(tmp) / "sample-report.xlsx"
            result = run_analysis(BASE_DIR, api_key="", output_path=output)
            self.assertEqual(result["total_skus"], 60)
            self.assertEqual(result["item_master"]["supplier"].nunique(), 6)
            self.assertEqual(len(result["analysis_results"]), 60)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()

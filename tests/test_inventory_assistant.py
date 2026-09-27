import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from inventory_assistant import (
    MissingNvidiaAPIKey,
    ask_inventory_assistant,
    build_analytical_context,
    calculate_stockout_risk,
    forecast_demand,
    get_nvidia_config,
    get_sku_summary,
    get_supplier_summary,
    rank_rop_gaps,
    rank_safety_stock_gaps,
    reset_inventory_session,
)


class FakeCompletions:
    def __init__(self):
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        message = SimpleNamespace(content="SKU A has a 20-unit Safety Stock gap.")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


class InventoryAssistantTests(unittest.TestCase):
    def setUp(self):
        self.items = pd.DataFrame(
            [
                {"sku": "A", "description": "Alpha", "supplier": "Supplier One"},
                {"sku": "B", "description": "Beta", "supplier": "Supplier Two"},
            ]
        )
        self.demand = pd.DataFrame(
            {
                "sku": ["A"] * 6 + ["B"] * 6,
                "week": list(range(1, 7)) * 2,
                "demand_qty": [10, 11, 12, 13, 14, 15, 20, 19, 18, 17, 16, 15],
            }
        )
        self.receipts = pd.DataFrame(
            {
                "sku": ["A", "A", "A", "B", "B", "B"],
                "po_number": ["1", "2", "3", "4", "5", "6"],
                "actual_lead_time_days": [10, 12, 14, 5, 5, 5],
            }
        )
        common = {
            "item_class": "A", "trend_pct": 20.0, "zero_demand_share": 0.0,
            "is_intermittent": False, "is_trending": False, "has_outliers": False,
            "lt_actual_mean_days": 12.0, "lt_actual_std_days": 2.0,
            "current_rop": 80, "recommended_rop": 100,
            "recent_weekly_mean": 12.5, "recent_weekly_std": 2.0,
        }
        self.results = [
            {**common, "sku": "A", "description": "Alpha", "supplier": "Supplier One",
             "current_safety_stock": 30, "recommended_safety_stock": 50},
            {**common, "sku": "B", "description": "Beta", "supplier": "Supplier Two",
             "item_class": "C", "trend_pct": -10.0, "lt_actual_mean_days": 5.0,
             "lt_actual_std_days": 0.0, "current_safety_stock": 60,
             "recommended_safety_stock": 40, "current_rop": 120, "recommended_rop": 90},
        ]

    def test_sku_retrieval_is_case_insensitive(self):
        self.assertEqual(get_sku_summary(self.results, "a")["description"], "Alpha")

    def test_supplier_join_and_aggregation(self):
        summary = get_supplier_summary(self.items, self.receipts, self.results)
        one = summary.loc[summary["supplier"] == "Supplier One"].iloc[0]
        self.assertEqual(one["receipt_count"], 3)
        self.assertEqual(one["average_lead_time_days"], 12)
        self.assertEqual(one["sku_count"], 1)

    def test_forecast_is_deterministic_and_bounded(self):
        result = forecast_demand(self.demand, "A", 4)
        self.assertEqual(len(result["forecast"]), 4)
        self.assertTrue((result["forecast"]["forecast_units"] >= 0).all())
        self.assertEqual(result["available_observations"], 6)

    def test_forecast_reports_insufficient_history(self):
        result = forecast_demand(pd.DataFrame({"sku": ["A"], "demand_qty": [1]}), "A")
        self.assertIn("At least 3", result["limitation"])

    def test_gap_rankings(self):
        self.assertEqual(rank_safety_stock_gaps(self.results).iloc[0]["sku"], "A")
        self.assertEqual(rank_rop_gaps(self.results).iloc[0]["sku"], "B")

    def test_stockout_risk_is_labeled_and_explainable(self):
        risk = calculate_stockout_risk(self.results)
        self.assertEqual(risk.iloc[0]["sku"], "A")
        self.assertIn(risk.iloc[0]["stockout_risk_level"], {"Low", "Moderate", "High"})
        self.assertGreaterEqual(risk.iloc[0]["stockout_risk_score"], 0)
        self.assertLessEqual(risk.iloc[0]["stockout_risk_score"], 100)

    def test_unsupported_historical_stockout_is_not_invented(self):
        _, _, metadata, direct = build_analytical_context(
            "How many times did SKU A stock out last year?",
            self.items, self.demand, self.receipts, self.results,
        )
        self.assertEqual(metadata["intent"], "unsupported")
        self.assertIn("not available", direct)

    def test_context_contains_only_relevant_ranked_rows(self):
        context, table, metadata, _ = build_analytical_context(
            "Top 1 Safety Stock increases",
            self.items, self.demand, self.receipts, self.results,
        )
        self.assertEqual(metadata["intent"], "safety_stock")
        self.assertEqual(len(table), 1)
        self.assertNotIn("Supplier Two", context)

    def test_missing_nvidia_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(MissingNvidiaAPIKey):
                get_nvidia_config()

    def test_mocked_nemotron_receives_grounded_context(self):
        client = FakeClient()
        answer = ask_inventory_assistant(
            "Why should Safety Stock for SKU A increase?",
            self.items, self.demand, self.receipts, self.results,
            api_key="test-key", client=client,
        )
        self.assertIn("20-unit", answer.text)
        request = client.chat.completions.last_request
        self.assertIn("PYTHON-GENERATED EVIDENCE", request["messages"][-1]["content"])
        self.assertNotIn("test-key", str(request))

    def test_session_reset_removes_analysis_and_chat_only(self):
        state = {"analysis_results": [1], "inventory_chat": [1], "unrelated": "keep"}
        reset_inventory_session(state)
        self.assertNotIn("analysis_results", state)
        self.assertNotIn("inventory_chat", state)
        self.assertEqual(state["unrelated"], "keep")


if __name__ == "__main__":
    unittest.main()

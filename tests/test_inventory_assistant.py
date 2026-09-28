import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from inventory_assistant import (
    MissingNvidiaAPIKey,
    _read_local_env,
    ask_inventory_assistant,
    build_analytical_context,
    calculate_stockout_risk,
    forecast_demand,
    get_nvidia_config,
    get_sku_summary,
    get_supplier_summary,
    plan_question,
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


class SequentialCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        content = self.contents.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class SequentialClient:
    def __init__(self, contents):
        self.chat = SimpleNamespace(completions=SequentialCompletions(contents))


class InventoryAssistantTests(unittest.TestCase):
    def setUp(self):
        self.items = pd.DataFrame(
            [
                {"sku": "A", "description": "Alpha", "supplier": "Supplier One",
                 "item_class": "A", "current_safety_stock": 30, "current_rop": 80},
                {"sku": "B", "description": "Beta", "supplier": "Supplier Two",
                 "item_class": "C", "current_safety_stock": 60, "current_rop": 120},
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
        self.assertEqual(metadata["intent"], "inventory_ranking")
        self.assertEqual(len(table), 1)
        self.assertNotIn("Supplier Two", context)

    def test_missing_nvidia_key(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "inventory_assistant.LOCAL_ENV_PATH", Path("missing-test.env")
        ):
            with self.assertRaises(MissingNvidiaAPIKey):
                get_nvidia_config()

    def test_local_env_configuration_is_loaded_without_mutating_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "NVIDIA_API_KEY=test-local-key\nNVIDIA_MODEL=test-model\n",
                encoding="utf-8",
            )
            values = _read_local_env(env_path)
            self.assertEqual(values["NVIDIA_MODEL"], "test-model")
            with patch.dict(os.environ, {}, clear=True), patch(
                "inventory_assistant.LOCAL_ENV_PATH", env_path
            ):
                config = get_nvidia_config()
                self.assertEqual(config["api_key"], "test-local-key")
                self.assertEqual(config["model"], "test-model")
                self.assertNotIn("NVIDIA_API_KEY", os.environ)

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
        self.assertEqual(
            answer.metadata["sources_used"],
            ["item_master", "demand_history", "receipt_history", "analysis_results"],
        )
        self.assertIn("Based on:", answer.text)

    def test_low_confidence_language_uses_validated_nemotron_plan(self):
        client = SequentialClient([
            '{"query_type":"ranking","entity_type":"sku",'
            '"metric":"stockout_risk","metrics":["stockout_risk"],'
            '"direction":"descending","limit":1,"conditions":[]}',
            "SKU A has the highest relative inventory risk.",
        ])
        answer = ask_inventory_assistant(
            "Show me the item with the greatest planning exposure.",
            self.items, self.demand, self.receipts, self.results,
            api_key="test-key", client=client,
        )
        self.assertEqual(answer.metadata["intent"], "stockout_risk")
        self.assertEqual(answer.metadata["evidence"]["question_plan"]["metric"], "stockout_risk")
        self.assertEqual(len(answer.table), 1)
        self.assertEqual(len(client.chat.completions.requests), 2)

    def test_source_selection_uses_the_minimum_authoritative_datasets(self):
        supplier = plan_question("What supplier provides SKU A?", self.items)
        demand = plan_question("What was demand for SKU A over the last 4 weeks?", self.items)
        receipts = plan_question("What is average actual lead time for SKU A?", self.items)
        safety = plan_question("Why should Safety Stock for SKU A increase?", self.items)
        self.assertEqual(supplier.required_sources, ["item_master"])
        self.assertEqual(demand.required_sources, ["demand_history"])
        self.assertEqual(receipts.required_sources, ["receipt_history"])
        self.assertEqual(
            safety.required_sources,
            ["item_master", "demand_history", "receipt_history", "analysis_results"],
        )

    def test_demand_context_uses_raw_history_not_analysis_summary(self):
        altered_results = [dict(row) for row in self.results]
        altered_results[0]["recent_weekly_mean"] = 9999
        context, table, metadata, _ = build_analytical_context(
            "What was demand for SKU A over the last 4 weeks?",
            self.items, self.demand, self.receipts, altered_results,
        )
        self.assertEqual(metadata["sources_used"], ["demand_history"])
        self.assertEqual(table["demand_qty"].tolist(), [12, 13, 14, 15])
        self.assertNotIn("9999", context)

    def test_follow_up_prefers_focused_sku_and_requeries_current_data(self):
        _, first_table, first_metadata, _ = build_analytical_context(
            "Top 1 Safety Stock increases",
            self.items, self.demand, self.receipts, self.results,
        )
        self.assertEqual(first_table.iloc[0]["sku"], "A")
        changed_demand = self.demand.copy()
        changed_demand.loc[changed_demand["sku"] == "A", "demand_qty"] = 999
        context, table, metadata, _ = build_analytical_context(
            "Why?",
            self.items, changed_demand, self.receipts, self.results,
            prior_metadata=first_metadata,
        )
        self.assertEqual(metadata["intent"], "sku_profile")
        self.assertEqual(table.iloc[0]["sku"], "A")
        self.assertEqual(table.iloc[0]["demand_recent_avg"], 999)
        self.assertIn("999", context)

    def test_explicit_unknown_sku_is_reported_instead_of_showing_portfolio(self):
        _, table, metadata, direct = build_analytical_context(
            "Tell me about SKU UNKNOWN-404",
            self.items, self.demand, self.receipts, self.results,
        )
        self.assertEqual(metadata["intent"], "unknown_entity")
        self.assertIsNone(table)
        self.assertIn("Unknown SKU", direct)

    def test_session_reset_removes_raw_analysis_cache_and_chat_together(self):
        state = {
            "item_master": [1], "demand_history": [1], "receipt_history": [1],
            "analysis_results": [1], "inventory_chat": [1], "assistant_metadata": {"focus_sku": "OLD"},
            "assistant_data_model": object(), "unrelated": "keep",
        }
        reset_inventory_session(state)
        for key in (
            "item_master", "demand_history", "receipt_history", "analysis_results",
            "inventory_chat", "assistant_metadata", "assistant_data_model",
        ):
            self.assertNotIn(key, state)
        self.assertEqual(state["unrelated"], "keep")


if __name__ == "__main__":
    unittest.main()

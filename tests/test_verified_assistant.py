import json
import unittest
from types import SimpleNamespace

from tests.test_verified_analytics import fixture_analytics
from evaluate_assistant import build_sample_analytics
from verified_inventory_assistant import (
    PlanningError,
    ask_verified_inventory_assistant,
    initial_conversation_context,
    normalize_conversation_context,
    resolve_verified_follow_up,
    validate_plan,
)


class ScriptedClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.chat = SimpleNamespace(completions=self)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = self.payloads.pop(0)
        if isinstance(value, Exception):
            raise value
        content = value if isinstance(value, str) else json.dumps(value)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class ValidatedPlanningTests(unittest.TestCase):
    def setUp(self):
        self.analytics = fixture_analytics()
        self.model = self.analytics.model

    def ask(self, payload, question, context=None):
        return ask_verified_inventory_assistant(
            question=question,
            data_model=self.model,
            client=ScriptedClient([payload]),
            model="scripted-model",
            prior_context=context,
        )

    def test_variability_question_cannot_treat_is_as_sku(self):
        payload = {"outcome": "execute", "steps": [{
            "operation": "rank_skus",
            "arguments": {"metric": "lead_time_variability", "direction": "descending", "limit": 1},
        }]}
        answer = self.ask(payload, "Which SKU is having the most lead-time variability?")
        self.assertEqual(answer.table.iloc[0]["sku"], "A")
        self.assertEqual(answer.evidence["coverage"]["evaluated_entity_count"], 2)
        self.assertNotIn("is", answer.metadata["current_skus"])

    def test_sku_to_po_to_specific_po_follow_up(self):
        ranking = self.ask({"outcome": "execute", "steps": [{
            "operation": "rank_skus",
            "arguments": {"metric": "lead_time_variability", "direction": "descending", "limit": 1},
        }]}, "Which SKU has the most lead-time variability?")
        po = self.ask({"outcome": "execute", "steps": [{
            "operation": "longest_receipt", "arguments": {"skus": "$context.current_skus"},
        }]}, "Which PO for this SKU has the longest lead time?", ranking.metadata)
        self.assertEqual(po.table.iloc[0]["po_number"], "PA-2")
        specific = self.ask({"outcome": "execute", "steps": [{
            "operation": "show_prior_result",
            "arguments": {"result_id": "$context.latest_result_id"},
        }]}, "Can you pick the specific PO please?", po.metadata)
        self.assertIn("PA-2", specific.text)
        self.assertEqual(specific.metadata["current_pos"], ["PA-2"])

    def test_undefined_threshold_gets_clarification(self):
        answer = self.ask(
            {"outcome": "clarify", "message": "What lead-time variability threshold should define highly inconsistent?"},
            "Find A items with increasing six-week demand and highly inconsistent delivery.",
        )
        self.assertIsNone(answer.table)
        self.assertEqual(answer.evidence["outcome"], "clarify")

    def test_malformed_plan_has_one_repair_then_stops(self):
        client = ScriptedClient(["not json", {"outcome": "execute", "steps": [{"operation": "delete"}]}])
        with self.assertRaisesRegex(PlanningError, "invalid requests twice"):
            ask_verified_inventory_assistant(
                question="Do something", data_model=self.model, client=client,
                model="scripted-model",
            )
        self.assertEqual(len(client.calls), 2)

    def test_provider_outage_does_not_execute_or_invent(self):
        client = ScriptedClient([RuntimeError("offline")])
        with self.assertRaisesRegex(PlanningError, "unavailable"):
            ask_verified_inventory_assistant(
                question="Which SKU is highest?", data_model=self.model,
                client=client, model="scripted-model",
            )

    def test_dataset_change_invalidates_structured_context(self):
        old = initial_conversation_context("old")
        old["current_skus"] = ["A"]
        old["prior_results"] = [{"result_id": "result-old"}]
        reset = normalize_conversation_context(old, "new")
        self.assertEqual(reset["current_skus"], [])
        self.assertEqual(reset["prior_results"], [])

    def test_specific_po_follow_up_clarifies_a_tied_prior_set(self):
        context = initial_conversation_context(self.analytics.fingerprint)
        context["current_pos"] = ["P1", "P2"]
        context["prior_results"] = [{"result_id": "result-tie"}]
        plan = resolve_verified_follow_up("Can you pick the specific PO?", context)
        self.assertEqual(plan.outcome, "clarify")
        self.assertIn("multiple", plan.message)

    def test_plan_rejects_unknown_arguments(self):
        with self.assertRaises(PlanningError):
            validate_plan({"outcome": "execute", "steps": [{
                "operation": "rank_skus", "arguments": {"metric": "mean_demand", "path": "x"},
            }]})

    def test_mixed_analysis_and_recommendation_uses_both_paths(self):
        client = ScriptedClient([
            {
                "outcome": "respond",
                "response_paths": ["analysis", "recommendation"],
                "knowledge_topics": ["lead_time_variability", "supplier_management"],
                "knowledge_request": None,
                "recommendation_request": "Suggest actions for the verified most-variable SKU.",
                "steps": [{
                    "operation": "rank_skus",
                    "arguments": {
                        "metric": "lead_time_variability", "direction": "descending", "limit": 1,
                    },
                }],
            },
            {
                "explanation": "Reducing variability requires isolating controllable process differences.",
                "possible_causes": ["Supplier queue-time inconsistency."],
                "proposed_actions": ["Review source receipt records with the supplier."],
                "caveat": "The cause is a hypothesis, not a finding from the uploaded data.",
            },
        ])
        answer = ask_verified_inventory_assistant(
            question="Which SKU is most variable and what should we do?",
            data_model=self.model,
            client=client,
            model="scripted-model",
        )
        self.assertEqual(answer.table.iloc[0]["sku"], "A")
        self.assertIn("Verified observations", answer.text)
        self.assertIn("Possible causes", answer.text)
        self.assertIn("Proposed actions", answer.text)
        self.assertEqual(answer.evidence["response_paths"], ["analysis", "recommendation"])


class ScreenshotRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analytics = build_sample_analytics()
        cls.model = cls.analytics.model

    def test_safety_stock_definition_needs_no_analytical_query(self):
        client = ScriptedClient([
            {
                "outcome": "respond",
                "response_paths": ["knowledge"],
                "knowledge_topics": ["safety_stock"],
                "knowledge_request": "Define safety stock, its purpose, and its main uncertainty drivers.",
                "recommendation_request": None,
                "steps": [],
            },
            {
                "explanation": (
                    "Safety stock is inventory held above expected cycle demand to protect against "
                    "uncertainty in demand and replenishment lead time."
                ),
                "possible_causes": [],
                "proposed_actions": [],
                "caveat": None,
            },
        ])
        answer = ask_verified_inventory_assistant(
            question="what is safety stock?",
            data_model=self.model,
            client=client,
            model="scripted-model",
        )
        self.assertIsNone(answer.table)
        self.assertIn("Safety stock", answer.text)
        self.assertIn("Inventory concept", answer.text)
        self.assertEqual(answer.evidence["response_paths"], ["knowledge"])
        self.assertEqual(answer.evidence["results"], [])
        self.assertEqual(len(client.calls), 2)

    def test_mat1019_variability_recommendation_uses_verified_context(self):
        ranking = ask_verified_inventory_assistant(
            question="Which SKU is having the most lead-time variability?",
            data_model=self.model,
            client=ScriptedClient([{
                "outcome": "execute", "steps": [{
                    "operation": "rank_skus",
                    "arguments": {
                        "metric": "lead_time_variability", "direction": "descending", "limit": 1,
                    },
                }],
            }]),
            model="scripted-model",
        )
        po = ask_verified_inventory_assistant(
            question="Which PO corresponds to the largest lead time for this SKU?",
            data_model=self.model,
            client=ScriptedClient([{
                "outcome": "execute", "steps": [{
                    "operation": "longest_receipt",
                    "arguments": {"skus": "$context.current_skus"},
                }],
            }]),
            model="scripted-model",
            prior_context=ranking.metadata,
        )
        client = ScriptedClient([
            {
                "outcome": "respond",
                "response_paths": ["recommendation"],
                "knowledge_topics": ["lead_time_variability", "supplier_management"],
                "knowledge_request": None,
                "recommendation_request": (
                    "Suggest ways to reduce lead-time variability for the prior verified SKU."
                ),
                "steps": [],
            },
            {
                "explanation": "Focus on isolating repeatable versus exceptional replenishment paths.",
                "possible_causes": [
                    "Supplier processing or queue-time inconsistency.",
                    "Different transport or expediting paths.",
                ],
                "proposed_actions": [
                    "Review the verified source receipts with procurement and the supplier.",
                    "Agree on lead-time definitions and monitor the same variability metric.",
                ],
                "caveat": "These causes are hypotheses and are not proven by the uploaded fields.",
            },
        ])
        answer = ask_verified_inventory_assistant(
            question="what are the suggested plans to reduce this variability?",
            data_model=self.model,
            client=client,
            model="scripted-model",
            prior_context=po.metadata,
        )
        self.assertIsNone(answer.table)
        self.assertIn("MAT-1019", answer.text)
        self.assertIn("PO-MAT-1019-018", answer.text)
        self.assertIn("Possible causes", answer.text)
        self.assertIn("not findings proven", answer.text)
        self.assertIn("Proposed actions", answer.text)
        self.assertNotIn("Are you looking for suggestions", answer.text)
        self.assertEqual(answer.metadata["current_skus"], ["MAT-1019"])
        self.assertEqual(answer.evidence["response_paths"], ["recommendation"])

    def test_ungrounded_recommendation_content_falls_back_safely(self):
        ranking = ask_verified_inventory_assistant(
            question="Which SKU is having the most lead-time variability?",
            data_model=self.model,
            client=ScriptedClient([{
                "outcome": "execute", "steps": [{
                    "operation": "rank_skus",
                    "arguments": {
                        "metric": "lead_time_variability", "direction": "descending", "limit": 1,
                    },
                }],
            }]),
            model="scripted-model",
        )
        client = ScriptedClient([
            {
                "outcome": "respond", "response_paths": ["recommendation"],
                "knowledge_topics": ["lead_time_variability"],
                "recommendation_request": "Recommend actions for the verified SKU.",
                "steps": [],
            },
            {
                "explanation": "The data proves another SKU is the cause.",
                "possible_causes": ["MAT-1002 caused the delay."],
                "proposed_actions": ["Order 999 units immediately."],
                "caveat": None,
            },
        ])
        answer = ask_verified_inventory_assistant(
            question="How can we reduce this variability?",
            data_model=self.model,
            client=client,
            model="scripted-model",
            prior_context=ranking.metadata,
        )
        self.assertNotIn("MAT-1002", answer.text)
        self.assertNotIn("999", answer.text)
        self.assertTrue(any("fallback" in warning for warning in answer.evidence["warnings"]))

    def test_general_knowledge_turn_preserves_verified_entity_context(self):
        ranking = ask_verified_inventory_assistant(
            question="Which SKU is having the most lead-time variability?",
            data_model=self.model,
            client=ScriptedClient([{
                "outcome": "execute", "steps": [{
                    "operation": "rank_skus",
                    "arguments": {
                        "metric": "lead_time_variability", "direction": "descending", "limit": 1,
                    },
                }],
            }]),
            model="scripted-model",
        )
        answer = ask_verified_inventory_assistant(
            question="What is safety stock?",
            data_model=self.model,
            client=ScriptedClient([
                {
                    "outcome": "respond", "response_paths": ["knowledge"],
                    "knowledge_topics": ["safety_stock"],
                    "knowledge_request": "Define safety stock.",
                    "steps": [],
                },
                {
                    "explanation": "Safety stock is a buffer for demand and lead-time uncertainty.",
                    "possible_causes": [], "proposed_actions": [], "caveat": None,
                },
            ]),
            model="scripted-model",
            prior_context=ranking.metadata,
        )
        self.assertEqual(answer.metadata["current_skus"], ["MAT-1019"])
        self.assertEqual(answer.evidence["verified_context_results"], [])


if __name__ == "__main__":
    unittest.main()

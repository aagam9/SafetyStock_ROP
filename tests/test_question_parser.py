import unittest

import pandas as pd

from inventory_assistant import plan_question


class NaturalLanguageQuestionParserTests(unittest.TestCase):
    def setUp(self):
        self.items = pd.DataFrame([
            {
                "sku": "MAT-1001", "description": "Motor Assembly",
                "supplier": "Supplier Alpha", "item_class": "A",
            },
            {
                "sku": "MAT-1002", "description": "Pump Assembly",
                "supplier": "Supplier Beta", "item_class": "B",
            },
        ])

    def assert_lead_variability_ranking(self, question):
        plan = plan_question(question, self.items)
        self.assertEqual(plan.intent, "receipt_ranking", question)
        self.assertEqual(plan.query_type, "ranking", question)
        self.assertEqual(plan.entity_type, "sku", question)
        self.assertEqual(plan.metric, "lead_time_variability", question)
        self.assertEqual(plan.direction, "descending", question)
        self.assertEqual(plan.limit, 1, question)
        self.assertEqual(plan.skus, [], question)
        self.assertEqual(plan.required_sources, ["receipt_history"], question)

    def test_lead_time_ranking_grammatical_variations(self):
        questions = [
            "Which SKU has the highest lead-time variability?",
            "Which SKU is having the most lead-time variability?",
            "Which item has the most inconsistent lead times?",
            "What product has the greatest variation in lead time?",
            "Which SKU shows the worst lead-time consistency?",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assert_lead_variability_ranking(question)

    def test_specific_sku_lead_time_variations(self):
        for question in (
            "What is the lead-time variability for MAT-1001?",
            "Tell me the lead time for SKU MAT-1001.",
            "How inconsistent are receipts for MAT-1001?",
        ):
            with self.subTest(question=question):
                plan = plan_question(question, self.items)
                self.assertEqual(plan.intent, "sku_receipts")
                self.assertEqual(plan.skus, ["MAT-1001"])
                self.assertEqual(plan.required_sources, ["receipt_history"])

    def test_demand_ranking_variations(self):
        cases = {
            "Which SKU has the most volatile demand?": 1,
            "What item has the highest demand variability?": 1,
            "Show me the top 5 products by demand volatility.": 5,
            "Which products have unstable demand?": 10,
        }
        for question, limit in cases.items():
            with self.subTest(question=question):
                plan = plan_question(question, self.items)
                self.assertEqual(plan.intent, "demand_ranking")
                self.assertEqual(plan.metric, "demand_variability")
                self.assertEqual(plan.direction, "descending")
                self.assertEqual(plan.limit, limit)

    def test_safety_stock_ranking_variations(self):
        cases = {
            "Which SKU has the largest Safety Stock gap?": 1,
            "Which products need the biggest increase in Safety Stock?": 10,
            "Show the top 10 items where recommended SS exceeds current SS.": 10,
        }
        for question, limit in cases.items():
            with self.subTest(question=question):
                plan = plan_question(question, self.items)
                self.assertEqual(plan.intent, "inventory_ranking")
                self.assertEqual(plan.metric, "safety_stock_gap")
                self.assertEqual(plan.limit, limit)

    def test_supplier_language_and_direction(self):
        highest = plan_question("Which supplier has the most inconsistent lead times?", self.items)
        lowest = plan_question("Which vendor has the lowest lead-time variability?", self.items)
        filtered = plan_question("Show Supplier Alpha SKUs with volatile demand.", self.items)
        risk = plan_question("Which Supplier Alpha item has the highest stockout risk?", self.items)

        self.assertEqual((highest.intent, highest.entity_type), ("supplier_overview", "supplier"))
        self.assertEqual(highest.direction, "descending")
        self.assertEqual(lowest.direction, "ascending")
        self.assertEqual(filtered.intent, "supplier_demand")
        self.assertEqual(filtered.suppliers, ["Supplier Alpha"])
        self.assertEqual(risk.intent, "stockout_risk")
        self.assertEqual(risk.filters["supplier"], ["Supplier Alpha"])

    def test_cross_file_compound_language(self):
        questions = [
            "Which A-class SKUs have high demand variability and unstable lead times?",
            "Which Supplier Alpha products have increasing demand and insufficient Safety Stock?",
            "Which products have both high lead times and large ROP gaps?",
        ]
        for question in questions:
            with self.subTest(question=question):
                plan = plan_question(question, self.items)
                self.assertEqual(plan.intent, "cross_file_risk")
                self.assertGreaterEqual(len(plan.required_sources), 2)

    def test_grammar_after_sku_is_never_an_identifier(self):
        grammar_words = [
            "is", "has", "having", "with", "that", "which", "shows",
            "appears", "needs", "should", "can", "does", "seems",
        ]
        for word in grammar_words:
            question = f"Which SKU {word} the highest lead-time variability?"
            with self.subTest(word=word):
                plan = plan_question(question, self.items)
                self.assertNotEqual(plan.intent, "unknown_entity")
                self.assertEqual(plan.skus, [])

    def test_unknown_identifier_requires_identifier_evidence(self):
        unknown = plan_question("Tell me about SKU MAT-9999", self.items)
        grammar = plan_question("Which SKU DOES have the highest demand?", self.items)
        self.assertEqual(unknown.intent, "unknown_entity")
        self.assertIn("MAT-9999", unknown.limitations[0])
        self.assertNotEqual(grammar.intent, "unknown_entity")

    def test_comparison_and_explicit_entity_regression(self):
        comparison = plan_question("Compare MAT-1001 and MAT-1002.", self.items)
        rop = plan_question("What is the recommended ROP for MAT-1001?", self.items)
        self.assertEqual(comparison.intent, "sku_comparison")
        self.assertEqual(comparison.skus, ["MAT-1001", "MAT-1002"])
        self.assertEqual(rop.intent, "sku_rop")
        self.assertEqual(rop.skus, ["MAT-1001"])

    def test_ambiguous_known_name_requests_clarification(self):
        items = pd.DataFrame([
            {"sku": "ALPHA", "description": "Alpha", "supplier": "Supplier One", "item_class": "A"},
            {"sku": "BETA", "description": "Beta", "supplier": "Supplier Alpha", "item_class": "B"},
        ])
        plan = plan_question("Show me Alpha", items)
        self.assertEqual(plan.intent, "clarification")
        self.assertTrue(plan.ambiguous_entities)
        explicit = plan_question("Show Supplier Alpha SKUs", items)
        self.assertEqual(explicit.intent, "supplier_skus")
        self.assertEqual(explicit.suppliers, ["Supplier Alpha"])
        self.assertEqual(explicit.skus, [])


if __name__ == "__main__":
    unittest.main()

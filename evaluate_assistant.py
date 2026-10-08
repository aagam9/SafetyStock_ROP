"""Reproducible deterministic benchmark and optional live-planner evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from inventory_analytics import InventoryAnalytics
from inventory_assistant import get_nvidia_config
from inventory_contracts import validate_inventory_frames
from inventory_data import InventoryDataModel
from safety_stock_agent import analyze_sku
from verified_inventory_assistant import ask_verified_inventory_assistant


BASE_DIR = Path(__file__).resolve().parent


def build_sample_analytics() -> InventoryAnalytics:
    """Build the analytical interface from bundled CSVs without model calls."""

    normalized, report = validate_inventory_frames(
        pd.read_csv(BASE_DIR / "item_master.csv"),
        pd.read_csv(BASE_DIR / "demand_history.csv"),
        pd.read_csv(BASE_DIR / "receipt_history.csv"),
    )
    items = normalized["item_master"]
    demand = normalized["demand_history"]
    receipts = normalized["receipt_history"]
    analysis_results = []
    for _, item in items.iterrows():
        sku = item["sku"]
        analysis_results.append(analyze_sku(
            item.to_dict(),
            demand.loc[demand["sku"] == sku, "demand_qty"].reset_index(drop=True),
            receipts.loc[receipts["sku"] == sku, "actual_lead_time_days"],
        ))
    model = InventoryDataModel.build(items, demand, receipts, analysis_results)
    return InventoryAnalytics(model, report.warnings)


def deterministic_benchmark(analytics: InventoryAnalytics) -> dict[str, Any]:
    """Compute expected identifiers independently of the planning provider."""

    variability = analytics.rank_skus("lead_time_variability", limit=5)
    mean_lead = analytics.rank_skus("mean_lead_time", limit=1)
    longest = analytics.longest_receipt()
    return {
        "dataset_fingerprint": analytics.fingerprint,
        "top_five_lead_time_variability": variability.rows,
        "variability_coverage": {
            "eligible": variability.eligible_entity_count,
            "evaluated": variability.evaluated_entity_count,
            "excluded": variability.excluded_count,
        },
        "highest_mean_lead_time": mean_lead.rows,
        "longest_individual_receipt": longest.rows,
        "mean_and_individual_winners_are_distinct": (
            mean_lead.rows[0]["sku"] != longest.rows[0]["sku"]
            if mean_lead.rows and longest.rows else None
        ),
    }


def _result_row_ids(answer: Any) -> dict[str, list[str]]:
    return {
        "skus": sorted({str(value) for value in answer.table.get("sku", []) if pd.notna(value)})
        if answer.table is not None and "sku" in answer.table else [],
        "pos": sorted({str(value) for value in answer.table.get("po_number", []) if pd.notna(value)})
        if answer.table is not None and "po_number" in answer.table else [],
    }


def live_benchmark(analytics: InventoryAnalytics, runs: int) -> dict[str, Any]:
    """Measure live semantic planning separately from deterministic correctness."""

    from openai import OpenAI

    config = get_nvidia_config()
    client = OpenAI(
        base_url=config["base_url"], api_key=config["api_key"], timeout=30.0, max_retries=1
    )
    expected_variable = analytics.rank_skus("lead_time_variability", limit=1)
    expected_skus = sorted(str(row["sku"]) for row in expected_variable.rows)
    expected_po = analytics.longest_receipt(expected_skus)
    expected_pos = sorted(str(row["po_number"]) for row in expected_po.rows)
    measured: list[dict[str, Any]] = []
    for run_number in range(1, max(int(runs), 1) + 1):
        context = None
        run: dict[str, Any] = {"run": run_number, "steps": []}
        questions = [
            "Which SKU is having the most lead-time variability?",
            "Which PO for this SKU has the longest lead time?",
            "Can you pick the specific PO please?",
        ]
        for index, question in enumerate(questions):
            try:
                answer = ask_verified_inventory_assistant(
                    question=question,
                    data_model=analytics.model,
                    client=client,
                    model=config["model"],
                    prior_context=context,
                    validation_warnings=analytics.validation_warnings,
                )
                context = answer.metadata
                ids = _result_row_ids(answer)
                expected = expected_skus if index == 0 else expected_pos
                actual = ids["skus"] if index == 0 else ids["pos"]
                run["steps"].append({
                    "question": question,
                    "outcome": answer.evidence.get("plan", {}).get("outcome"),
                    "operations": [
                        step["operation"] for step in answer.evidence.get("plan", {}).get("steps", [])
                    ],
                    "expected_identifiers": expected,
                    "actual_identifiers": actual,
                    "exact_identifier_match": actual == expected,
                    "grounded_result_id": answer.evidence.get("result_id"),
                })
            except Exception as exc:
                run["steps"].append({
                    "question": question,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "exact_identifier_match": False,
                })
                break
        if context and len(run["steps"]) == 3:
            recommendation_question = "How can we reduce this variability?"
            try:
                answer = ask_verified_inventory_assistant(
                    question=recommendation_question,
                    data_model=analytics.model,
                    client=client,
                    model=config["model"],
                    prior_context=context,
                    validation_warnings=analytics.validation_warnings,
                )
                context = answer.metadata
                paths = answer.evidence.get("response_paths", [])
                run["steps"].append({
                    "question": recommendation_question,
                    "expected_response_path": "recommendation",
                    "actual_response_paths": paths,
                    "path_match": "recommendation" in paths,
                    "expected_context_skus": expected_skus,
                    "actual_context_skus": context.get("current_skus", []),
                    "context_match": context.get("current_skus", []) == expected_skus,
                    "used_verified_context_results": answer.evidence.get(
                        "verified_context_results", []
                    ),
                    "conceptual_fallback": any(
                        "fallback" in warning for warning in answer.evidence.get("warnings", [])
                    ),
                })
            except Exception as exc:
                run["steps"].append({
                    "question": recommendation_question,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "path_match": False,
                })
        measured.append(run)
    independent_cases: list[dict[str, Any]] = []
    cases = [
        {
            "question": "Show the top five SKUs by lead-time variability.",
            "expected_skus": [
                str(row["sku"]) for row in analytics.rank_skus(
                    "lead_time_variability", limit=5
                ).rows
            ],
            "expected_outcome": "respond",
        },
        {
            "question": (
                "Which SKU has the highest mean lead time, and which PO has the "
                "longest individual lead time?"
            ),
            "expected_skus": sorted({
                str(analytics.rank_skus("mean_lead_time", limit=1).rows[0]["sku"]),
                str(analytics.longest_receipt().rows[0]["sku"]),
            }),
            "expected_pos": [str(analytics.longest_receipt().rows[0]["po_number"])],
            "expected_outcome": "respond",
        },
        {
            "question": (
                "Which Class A SKUs had increasing demand over the last six weeks and "
                "highly inconsistent supplier delivery?"
            ),
            "expected_outcome": "clarify",
        },
        {
            "question": "Which supplier had the worst OTIF last quarter?",
            "expected_outcome": "unsupported",
        },
        {
            "question": "What is safety stock?",
            "expected_outcome": "respond",
            "expected_response_path": "knowledge",
        },
    ]
    for case in cases:
        try:
            answer = ask_verified_inventory_assistant(
                question=case["question"], data_model=analytics.model, client=client,
                model=config["model"], validation_warnings=analytics.validation_warnings,
            )
            ids = _result_row_ids(answer)
            outcome = answer.evidence.get("plan", {}).get("outcome") or answer.evidence.get("outcome")
            record = {
                "question": case["question"],
                "expected_outcome": case["expected_outcome"],
                "actual_outcome": outcome,
                "outcome_match": outcome == case["expected_outcome"],
                "actual_skus": ids["skus"],
                "actual_pos": ids["pos"],
            }
            if "expected_response_path" in case:
                record["expected_response_path"] = case["expected_response_path"]
                record["actual_response_paths"] = answer.evidence.get("response_paths", [])
                record["response_path_match"] = (
                    case["expected_response_path"] in answer.evidence.get("response_paths", [])
                )
            if "expected_skus" in case:
                record["expected_skus"] = sorted(case["expected_skus"])
                record["sku_set_match"] = ids["skus"] == sorted(case["expected_skus"])
            if "expected_pos" in case:
                record["expected_pos"] = sorted(case["expected_pos"])
                record["po_set_match"] = ids["pos"] == sorted(case["expected_pos"])
            independent_cases.append(record)
        except Exception as exc:
            independent_cases.append({
                "question": case["question"], "error_type": type(exc).__name__,
                "error": str(exc), "outcome_match": False,
            })
    return {
        "provider": "NVIDIA OpenAI-compatible endpoint",
        "model": config["model"],
        "base_url": config["base_url"],
        "runs": measured,
        "independent_cases": independent_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run optional live NVIDIA planning checks.")
    parser.add_argument("--runs", type=int, default=2, help="Repeated live runs (default: 2).")
    args = parser.parse_args()
    analytics = build_sample_analytics()
    report: dict[str, Any] = {"deterministic": deterministic_benchmark(analytics)}
    if args.live:
        report["live"] = live_benchmark(analytics, args.runs)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

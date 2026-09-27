#!/usr/bin/env python3
"""
Safety Stock / Reorder Point Drift Agent
=========================================
Scans a material master + demand/receipt history, recalculates what safety
stock and ROP *should* be based on actual demand variability and actual
supplier lead-time performance, flags SKUs where current parameters have
drifted from that, and uses Claude to make the judgment call on ambiguous
cases (trend, seasonality, outliers) that a formula can't resolve on its own.

Output: safety_stock_review.xlsx — a prioritized action list + portfolio summary.
"""

import os
import re
import sys
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from anthropic import Anthropic

# ── Secrets ────────────────────────────────────────────────────────────────
# settings.json next to this script: {"ANTHROPIC_API_KEY": "..."}
# or: export ANTHROPIC_API_KEY="..."
def _load_api_key() -> str | None:
    settings_path = Path(__file__).parent / "settings.json"
    key = None
    if settings_path.exists():
        try:
            key = json.load(open(settings_path)).get("ANTHROPIC_API_KEY")
        except Exception:
            pass
    return os.getenv("ANTHROPIC_API_KEY", key)

API_KEY = _load_api_key()
CLAUDE_MODEL = "claude-sonnet-4-5"

# ── Config ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUT_PATH = BASE_DIR / "safety_stock_review.xlsx"

DRIFT_THRESHOLD_PCT   = 0.25   # flag if recommended SS differs from current by >25%
LT_DRIFT_THRESHOLD    = 0.20   # flag if actual lead time differs from assumed by >20%
RECENT_WINDOW_WEEKS   = 26     # "recent" demand window for trend comparison
OUTLIER_Z             = 3.0    # weekly demand z-score to call a point an outlier

CLASS_Z = {"A": 2.05, "B": 1.65, "C": 1.28}   # 98% / 95% / 90% service levels


# ── Statistics engine ──────────────────────────────────────────────────────
def analyze_sku(item: dict, demand: pd.Series, lead_times: pd.Series) -> dict:
    """Compute actual demand/lead-time stats and the statistically-recommended
    safety stock & ROP for one SKU. Returns a dict of everything downstream
    steps need — including signals a pure threshold can't fully judge."""

    weekly = demand.values.astype(float)
    weekly_no_outliers = weekly.copy()
    mu, sigma = weekly.mean(), weekly.std(ddof=1) if len(weekly) > 1 else 0.0
    outlier_mask = np.zeros_like(weekly, dtype=bool)
    if sigma > 0:
        z_scores = (weekly - mu) / sigma
        outlier_mask = np.abs(z_scores) > OUTLIER_Z
        weekly_no_outliers[outlier_mask] = mu  # winsorize for a "clean" comparison series

    recent = weekly[-RECENT_WINDOW_WEEKS:]
    older = weekly[:-RECENT_WINDOW_WEEKS] if len(weekly) > RECENT_WINDOW_WEEKS else weekly

    recent_mean, recent_std = recent.mean(), recent.std(ddof=1) if len(recent) > 1 else 0.0
    older_mean = older.mean() if len(older) else recent_mean

    # crude trend flag: recent period materially different from prior period
    trend_pct = (recent_mean - older_mean) / older_mean if older_mean > 0 else 0.0
    is_trending = abs(trend_pct) > 0.25

    # intermittency: share of zero-demand weeks
    zero_share = float((weekly == 0).mean())
    is_intermittent = zero_share > 0.30

    has_outliers = bool(outlier_mask.any())

    lt_actual_mean = float(lead_times.mean())
    lt_actual_std = float(lead_times.std(ddof=1)) if len(lead_times) > 1 else 0.0
    lt_assumed = float(item["assumed_lead_time_days"])
    lt_drift_pct = (lt_actual_mean - lt_assumed) / lt_assumed if lt_assumed else 0.0

    # Convert weekly demand stats to daily for the SS formula (assumes 7-day weeks)
    d_daily_mean = recent_mean / 7
    d_daily_std = recent_std / 7
    lt_days_mean = lt_actual_mean
    lt_days_std = lt_actual_std   # already in days — no conversion needed

    z = CLASS_Z.get(item["item_class"], 1.65)
    # Combined demand + lead-time variability formula
    variance_term = lt_days_mean * (d_daily_std ** 2) + (d_daily_mean ** 2) * (lt_days_std ** 2)
    recommended_ss = z * np.sqrt(max(variance_term, 0))
    recommended_rop = d_daily_mean * lt_days_mean + recommended_ss

    current_ss = float(item["current_safety_stock"])
    ss_drift_pct = (recommended_ss - current_ss) / current_ss if current_ss > 0 else 1.0

    flagged = (
        abs(ss_drift_pct) > DRIFT_THRESHOLD_PCT
        or abs(lt_drift_pct) > LT_DRIFT_THRESHOLD
        or is_trending
        or is_intermittent
        or has_outliers
    )

    analysis_flags = []
    if abs(ss_drift_pct) > DRIFT_THRESHOLD_PCT:
        analysis_flags.append("material safety-stock gap")
    if abs(lt_drift_pct) > LT_DRIFT_THRESHOLD:
        analysis_flags.append("lead-time drift")
    if is_trending:
        analysis_flags.append("demand trend")
    if is_intermittent:
        analysis_flags.append("intermittent demand")
    if has_outliers:
        analysis_flags.append("demand outlier")

    return {
        **item,
        "recent_weekly_mean": round(recent_mean, 1),
        "recent_weekly_std": round(recent_std, 1),
        "trend_pct": round(trend_pct * 100, 1),
        "is_trending": is_trending,
        "zero_demand_share": round(zero_share * 100, 1),
        "is_intermittent": is_intermittent,
        "has_outliers": has_outliers,
        "service_level_z": z,
        "lt_actual_mean_days": round(lt_actual_mean, 1),
        "lt_actual_std_days": round(lt_actual_std, 1),
        "lt_drift_pct": round(lt_drift_pct * 100, 1),
        "recommended_safety_stock": round(recommended_ss),
        "recommended_rop": round(recommended_rop),
        "ss_drift_pct": round(ss_drift_pct * 100, 1),
        "flagged": flagged,
        "analysis_flags": analysis_flags,
        "demand_observations": int(len(demand)),
        "receipt_observations": int(len(lead_times)),
        "dollar_impact": round(abs(recommended_ss - current_ss) * item["unit_cost"], 2),
    }


# ── Claude reasoning step (only called for flagged SKUs) ──────────────────
_AGENT_PROMPT = """\
You are a senior inventory planning analyst reviewing a safety stock / \
reorder point recommendation that a statistical model flagged for human-level \
judgment. Statistics alone can't tell whether a signal is a real, durable shift \
or a distortion (e.g. one bulk order, a short blip, thin data). Decide.

SKU: {sku} — {description} (Class {item_class})
Unit cost: ${unit_cost}

Current safety stock: {current_safety_stock} | Current ROP: {current_rop}
Statistically recommended safety stock: {recommended_safety_stock} \
(drift vs current: {ss_drift_pct}%)
Statistically recommended ROP: {recommended_rop}

Recent 26-week avg weekly demand: {recent_weekly_mean} (std {recent_weekly_std})
Demand trend vs prior period: {trend_pct}% (flagged as trending: {is_trending})
Share of zero-demand weeks: {zero_demand_share}% (flagged intermittent: {is_intermittent})
Outlier demand week(s) detected: {has_outliers}
Assumed lead time: {assumed_lead_time_days} days | Actual avg lead time: \
{lt_actual_mean_days} days (drift: {lt_drift_pct}%)

Decide:
1. action: one of "increase_ss", "decrease_ss", "hold_no_change", "manual_review_needed"
   (use manual_review_needed only if the data is genuinely too ambiguous/thin to trust \
either the current or recommended number — e.g. an obvious one-off outlier still \
distorting the recommendation, or too little history)
2. confidence: "high", "medium", or "low"
3. risk_if_ignored: one short phrase, e.g. "stockout risk on lead-time slippage" or \
"tying up cash in excess stock" or "low risk either way"
4. rationale: 1-2 sentences, plain language, referencing the specific signal(s) that drove \
your decision

Output ONLY valid JSON, no markdown fences, no prose:
{{"action": "...", "confidence": "...", "risk_if_ignored": "...", "rationale": "..."}}
"""


def _call_claude(client: Anthropic, sku_data: dict) -> dict:
    prompt = _AGENT_PROMPT.format(**sku_data)
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw.strip())
    return json.loads(raw)


def _rule_based_fallback(sku_data: dict) -> dict:
    """Used only if no API key is configured — a transparent, simpler stand-in
    so the pipeline still runs end-to-end without an LLM."""
    if sku_data["has_outliers"] and abs(sku_data["ss_drift_pct"]) > 50:
        return {"action": "manual_review_needed", "confidence": "low",
                "risk_if_ignored": "recommendation may be skewed by an outlier order",
                "rationale": "An outlier demand week was detected and the drift is large; "
                             "needs a human to confirm before resetting the parameter."}
    if sku_data["ss_drift_pct"] > DRIFT_THRESHOLD_PCT * 100:
        return {"action": "increase_ss", "confidence": "medium",
                "risk_if_ignored": "stockout risk",
                "rationale": "Recommended safety stock exceeds current setting beyond threshold."}
    if sku_data["ss_drift_pct"] < -DRIFT_THRESHOLD_PCT * 100:
        return {"action": "decrease_ss", "confidence": "medium",
                "risk_if_ignored": "tying up cash in excess stock",
                "rationale": "Current safety stock is materially above statistical need."}
    return {"action": "hold_no_change", "confidence": "medium",
            "risk_if_ignored": "low risk either way",
            "rationale": "Other flags (trend/intermittency/lead time) present but drift is modest."}


def get_agent_recommendation(client, sku_data: dict,
                             progress_callback: Callable[[str], None] | None = None) -> dict:
    if client is None:
        return _rule_based_fallback(sku_data)
    for attempt in range(3):
        try:
            return _call_claude(client, sku_data)
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                message = f"    [Claude error after retries — using rule-based fallback] {e}"
                if progress_callback:
                    progress_callback(message)
                else:
                    print(message)
                return _rule_based_fallback(sku_data)


# ── Excel output ────────────────────────────────────────────────────────────
HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF")
BODY_FONT = Font(name="Arial", size=10)


def _style_header(ws, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")


def write_excel(review_queue: list[dict], all_results: list[dict],
                out_path: str | Path = OUT_PATH):
    wb = openpyxl.Workbook()

    # Sheet 1 — Review Queue (flagged SKUs, agent-reasoned)
    ws1 = wb.active
    ws1.title = "Review Queue"
    cols1 = ["sku", "description", "supplier", "item_class", "current_safety_stock",
             "recommended_safety_stock", "ss_drift_pct", "current_rop",
             "recommended_rop", "lt_drift_pct", "dollar_impact",
             "action", "confidence", "risk_if_ignored", "rationale"]
    headers1 = ["SKU", "Description", "Supplier", "Class", "Current SS", "Recommended SS",
                "SS Drift %", "Current ROP", "Recommended ROP", "LT Drift %",
                "$ Impact", "Agent Action", "Confidence", "Risk if Ignored", "Rationale"]
    ws1.append(headers1)
    for row in review_queue:
        ws1.append([row.get(c, "") for c in cols1])
    _style_header(ws1, len(headers1))
    widths1 = [12, 24, 20, 7, 11, 15, 10, 12, 15, 10, 11, 18, 11, 30, 50]
    for i, w in enumerate(widths1, 1):
        ws1.column_dimensions[get_column_letter(i)].width = w
    for r in range(2, ws1.max_row + 1):
        for c in range(1, len(headers1) + 1):
            ws1.cell(row=r, column=c).font = BODY_FONT
            ws1.cell(row=r, column=c).alignment = Alignment(wrap_text=(c == 15), vertical="top")

    # Sheet 2 — Full Portfolio Scan
    ws2 = wb.create_sheet("Portfolio Scan")
    cols2 = ["sku", "description", "supplier", "item_class", "current_safety_stock",
             "recommended_safety_stock", "ss_drift_pct", "current_rop", "recommended_rop",
             "recent_weekly_mean", "recent_weekly_std", "lt_actual_mean_days",
             "lt_actual_std_days", "lt_drift_pct",
             "is_trending", "is_intermittent", "has_outliers", "flagged"]
    headers2 = ["SKU", "Description", "Supplier", "Class", "Current SS", "Recommended SS",
                "SS Drift %", "Current ROP", "Recommended ROP", "Avg Weekly Demand",
                "Demand Std Dev", "Avg LT Days", "LT Std Dev", "LT Drift %", "Trending?", "Intermittent?",
                "Outliers?", "Flagged for Review?"]
    ws2.append(headers2)
    for row in all_results:
        ws2.append([row.get(c, "") for c in cols2])
    _style_header(ws2, len(headers2))
    widths2 = [12, 24, 20, 7, 11, 15, 10, 12, 15, 16, 15, 12, 12, 10, 10, 12, 10, 18]
    for i, w in enumerate(widths2, 1):
        ws2.column_dimensions[get_column_letter(i)].width = w
    for r in range(2, ws2.max_row + 1):
        for c in range(1, len(headers2) + 1):
            ws2.cell(row=r, column=c).font = BODY_FONT

    wb.save(Path(out_path))


# ── Pipeline / Main ───────────────────────────────────────────────────────
class SafetyStockInputError(ValueError):
    """Raised when an input folder or CSV does not match the required shape."""


_REQUIRED_COLUMNS = {
    "item_master.csv": {
        "sku", "description", "supplier", "item_class", "assumed_lead_time_days",
        "current_safety_stock", "current_rop", "unit_cost",
    },
    "demand_history.csv": {"sku", "week", "demand_qty"},
    "receipt_history.csv": {"sku", "po_number", "actual_lead_time_days"},
}

_NUMERIC_COLUMNS = {
    "item_master.csv": {
        "assumed_lead_time_days", "current_safety_stock", "current_rop", "unit_cost",
    },
    "demand_history.csv": {"demand_qty"},
    "receipt_history.csv": {"actual_lead_time_days"},
}


def _load_input_data(data_dir: Path):
    if not data_dir.is_dir():
        raise SafetyStockInputError(f"Input folder does not exist: {data_dir}")

    frames = {}
    for filename, required_columns in _REQUIRED_COLUMNS.items():
        path = data_dir / filename
        if not path.is_file():
            raise SafetyStockInputError(f"Missing required file: {filename}\nExpected in: {data_dir}")
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError,
                OSError) as exc:
            raise SafetyStockInputError(f"Could not read {filename}: {exc}") from exc

        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise SafetyStockInputError(
                f"{filename} is missing required column(s): {', '.join(missing)}"
            )
        if frame.empty:
            raise SafetyStockInputError(f"{filename} contains no data rows.")
        for column in _NUMERIC_COLUMNS[filename]:
            try:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise SafetyStockInputError(
                    f"{filename} column '{column}' contains a non-numeric value."
                ) from exc
        if frame["sku"].isna().any() or frame["sku"].astype(str).str.strip().eq("").any():
            raise SafetyStockInputError(f"{filename} contains a missing or blank SKU.")
        frame["sku"] = frame["sku"].astype(str).str.strip()
        frames[filename] = frame

    items = frames["item_master.csv"]
    demand = frames["demand_history.csv"]
    receipts = frames["receipt_history.csv"]

    if items["supplier"].isna().any() or items["supplier"].astype(str).str.strip().eq("").any():
        raise SafetyStockInputError("item_master.csv contains a missing or blank supplier.")
    items["supplier"] = items["supplier"].astype(str).str.strip()
    if items[list(_NUMERIC_COLUMNS["item_master.csv"])].isna().any().any():
        raise SafetyStockInputError("item_master.csv contains a null numeric calculation value.")
    if items["sku"].duplicated().any():
        duplicates = sorted(items.loc[items["sku"].duplicated(False), "sku"].unique())
        raise SafetyStockInputError(
            "item_master.csv must contain one row per SKU. Duplicate SKU(s): "
            + ", ".join(duplicates[:10])
        )
    if demand[["sku", "week"]].duplicated().any():
        raise SafetyStockInputError("demand_history.csv contains duplicate SKU/week rows.")
    if receipts["po_number"].isna().any() or receipts["po_number"].astype(str).str.strip().eq("").any():
        raise SafetyStockInputError("receipt_history.csv contains a missing or blank po_number.")
    if receipts["po_number"].astype(str).duplicated().any():
        raise SafetyStockInputError("receipt_history.csv contains duplicate po_number values.")
    if demand["demand_qty"].isna().any() or (demand["demand_qty"] < 0).any():
        raise SafetyStockInputError("demand_history.csv demand_qty must be non-null and nonnegative.")
    if receipts["actual_lead_time_days"].isna().any() or (receipts["actual_lead_time_days"] <= 0).any():
        raise SafetyStockInputError(
            "receipt_history.csv actual_lead_time_days must be non-null and positive."
        )
    week_text = demand["week"].astype(str).str.strip()
    numeric_week = pd.to_numeric(week_text, errors="coerce")
    if numeric_week.isna().any():
        date_week = pd.to_datetime(week_text, errors="coerce")
        if date_week.isna().any():
            raise SafetyStockInputError(
                "demand_history.csv column 'week' must contain valid week numbers or dates."
            )

    master_skus = set(items["sku"])
    for filename, frame in (("demand_history.csv", demand), ("receipt_history.csv", receipts)):
        unknown = sorted(set(frame["sku"]) - master_skus)
        missing_history = sorted(master_skus - set(frame["sku"]))
        if unknown:
            raise SafetyStockInputError(
                f"{filename} contains SKU(s) absent from item_master.csv: " + ", ".join(unknown[:10])
            )
        if missing_history:
            raise SafetyStockInputError(
                f"{filename} has no history for item-master SKU(s): "
                + ", ".join(missing_history[:10])
            )

    return items, demand, receipts


def run_analysis(data_dir: str | Path = BASE_DIR, api_key: str | None = None,
                 progress_callback: Callable[[str], None] | None = None,
                 output_path: str | Path | None = None) -> dict:
    """Run the analysis pipeline and return structured results.

    ``progress_callback`` receives complete log lines, making this safe to call
    from a GUI worker thread. If omitted, progress is printed as before.
    A supplied API key is used only for this call and is never stored or logged.
    """
    report = progress_callback or print
    data_dir = Path(data_dir)
    output_path = Path(output_path) if output_path else data_dir / OUT_PATH.name

    report("=" * 60)
    report("Safety Stock / ROP Drift Agent — Starting")
    report("=" * 60)

    session_api_key = api_key.strip() if api_key and api_key.strip() else API_KEY
    client = None
    if session_api_key:
        client = Anthropic(api_key=session_api_key)
        report(f"\n[0] Claude reasoning: enabled ({CLAUDE_MODEL})")
    else:
        report("\n[0] No ANTHROPIC_API_KEY found — running with rule-based fallback "
               "reasoning instead of Claude. Add settings.json or env var to enable "
               "full agent reasoning.")

    report("\n[1] Loading item master, demand history, receipt history...")
    items, demand_hist, receipt_hist = _load_input_data(data_dir)
    report(f"    {len(items)} SKUs | {len(demand_hist)} demand rows | "
           f"{len(receipt_hist)} PO receipts")

    report("\n[2] Running statistical drift analysis on full portfolio...")
    all_results = []
    for _, item in items.iterrows():
        item = item.to_dict()
        sku = item["sku"]
        demand = demand_hist.loc[demand_hist["sku"] == sku, "demand_qty"].reset_index(drop=True)
        lts = receipt_hist.loc[receipt_hist["sku"] == sku, "actual_lead_time_days"]
        result = analyze_sku(item, demand, lts)
        all_results.append(result)

    flagged = [r for r in all_results if r["flagged"]]
    report(f"    Scanned: {len(all_results)}  |  Flagged for review: {len(flagged)}"
           f"  |  Auto-cleared (params still sound): {len(all_results) - len(flagged)}")

    report(f"\n[3] Escalating {len(flagged)} flagged SKUs to agent reasoning...")
    review_queue = []
    for i, sku_data in enumerate(flagged, 1):
        status = (f"    [{i}/{len(flagged)}] {sku_data['sku']} — "
                  f"SS drift {sku_data['ss_drift_pct']:+.1f}%, "
                  f"LT drift {sku_data['lt_drift_pct']:+.1f}%...")
        rec = get_agent_recommendation(client, sku_data, report)
        report(f"{status} -> {rec['action']} ({rec['confidence']})")
        review_queue.append({**sku_data, **rec})

    # Sort by dollar impact, but push manual_review_needed / low confidence up
    # regardless of dollar size — those need eyes first even if the $ is small.
    def priority_key(r):
        urgent = r["action"] == "manual_review_needed" or r["confidence"] == "low"
        return (not urgent, -r["dollar_impact"])
    review_queue.sort(key=priority_key)

    report(f"\n[4] Writing results to {output_path.name}...")
    write_excel(review_queue, all_results, output_path)

    # Summary
    action_counts = {}
    for r in review_queue:
        action_counts[r["action"]] = action_counts.get(r["action"], 0) + 1
    total_dollar_impact = sum(r["dollar_impact"] for r in review_queue)
    analysis_summary = {
        "total_skus": len(all_results),
        "flagged_count": len(flagged),
        "auto_cleared_count": len(all_results) - len(flagged),
        "total_dollar_impact": total_dollar_impact,
        "action_counts": action_counts,
    }

    report("\n" + "=" * 60)
    report("SUMMARY")
    report("=" * 60)
    report(f"  Total SKUs scanned          : {len(all_results)}")
    report(f"  Flagged for review          : {len(flagged)}")
    report(f"  Total estimated $ at stake  : ${total_dollar_impact:,.0f}")
    report("\n  Recommended actions:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        report(f"    {action:<24}: {count}")
    report("\n  Top 5 by $ impact:")
    top5 = sorted(review_queue, key=lambda r: -r["dollar_impact"])[:5]
    for r in top5:
        report(f"    {r['sku']:<10} {r['description'][:30]:<32} "
               f"${r['dollar_impact']:>8,.0f}  {r['action']}")
    report(f"\nDone. Open {output_path.name} — start with the Review Queue tab.")

    return {
        "total_skus": len(all_results),
        "flagged_count": len(flagged),
        "auto_cleared_count": len(all_results) - len(flagged),
        "total_dollar_impact": total_dollar_impact,
        "action_counts": action_counts,
        "review_queue": review_queue,
        "all_results": all_results,
        "analysis_results": all_results,
        "analysis_summary": analysis_summary,
        "item_master": items.copy(),
        "demand_history": demand_hist.copy(),
        "receipt_history": receipt_hist.copy(),
        "output_path": output_path,
    }


def main():
    run_analysis()


if __name__ == "__main__":
    main()

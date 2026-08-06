#!/usr/bin/env python3
"""Business-user Streamlit interface for the Safety Stock / ROP Drift Agent."""

import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from safety_stock_agent import BASE_DIR, SafetyStockInputError, run_analysis


APP_TITLE = "Safety Stock / ROP Drift Review"
INPUT_FILES = ("item_master.csv", "demand_history.csv", "receipt_history.csv")
CALCULATION_HELP = """
**Calculation used**

- Safety stock: `z × √(L × σd² + d̄² × σL²)`
- Reorder point: `d̄ × L + safety stock`

- `d̄` and `σd`: mean and standard deviation from the most recent 26 weekly
  demand rows, converted to daily values by dividing by 7.
- `L` and `σL`: mean and standard deviation of actual receipt lead times in
  calendar days.
- Service assumptions: Class A uses `z = 2.05` (about 98%), Class B uses
  `z = 1.65` (about 95%), and Class C uses `z = 1.28` (about 90%). An unknown
  class defaults to `z = 1.65`.
- The service-level interpretation assumes an approximately normal demand
  distribution and reasonably stable, independent demand and lead-time variation.
"""
ACTION_LABELS = (
    ("increase_ss", "Increase"),
    ("decrease_ss", "Decrease"),
    ("hold_no_change", "Hold"),
    ("manual_review_needed", "Manual review"),
)

FILE_GUIDE = pd.DataFrame(
    [
        {
            "Exact filename": "item_master.csv",
            "One row per": "SKU",
            "Required columns": (
                "sku, description, item_class, assumed_lead_time_days, "
                "current_safety_stock, current_rop, unit_cost"
            ),
        },
        {
            "Exact filename": "demand_history.csv",
            "One row per": "SKU per week",
            "Required columns": "sku, week, demand_qty",
        },
        {
            "Exact filename": "receipt_history.csv",
            "One row per": "Received purchase order",
            "Required columns": "sku, po_number, actual_lead_time_days",
        },
    ]
)


def _deployment_api_key():
    """Read a deployed secret without requiring a local secrets file."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (FileNotFoundError, KeyError):
        return None


def _stage_inputs(run_dir: Path, source: str, uploads: dict):
    for filename in INPUT_FILES:
        destination = run_dir / filename
        if source == "Use bundled sample data":
            shutil.copyfile(BASE_DIR / filename, destination)
        else:
            destination.write_bytes(uploads[filename].getvalue())


def _progress_callback(log_placeholder, progress_bar):
    messages = []

    def report(message: str):
        messages.append(message)
        log_placeholder.code("\n".join(messages), language=None)

        if "[1] Loading" in message:
            progress_bar.progress(10, text="Loading and validating files")
        elif "[2] Running" in message:
            progress_bar.progress(25, text="Scanning the portfolio")
        elif "Scanned:" in message:
            progress_bar.progress(45, text="Portfolio scan complete")
        elif "[3] Escalating" in message:
            progress_bar.progress(50, text="Reviewing flagged SKUs")
        elif "[4] Writing" in message:
            progress_bar.progress(95, text="Creating the Excel report")
        elif message.startswith("Done.") or "\nDone." in message:
            progress_bar.progress(100, text="Analysis complete")
        else:
            escalation = re.search(r"\[(\d+)/(\d+)\]", message)
            if escalation and int(escalation.group(2)):
                current, total = map(int, escalation.groups())
                percent = 50 + round(40 * current / total)
                progress_bar.progress(
                    percent,
                    text=f"Reviewing flagged SKU {current} of {total}",
                )

    return report


def _save_run_result(result: dict):
    st.session_state["last_report"] = Path(result["output_path"]).read_bytes()
    st.session_state["last_summary"] = {
        "total_skus": result["total_skus"],
        "flagged_count": result["flagged_count"],
        "total_dollar_impact": result["total_dollar_impact"],
        "action_counts": dict(result["action_counts"]),
    }


def _render_summary():
    summary = st.session_state.get("last_summary")
    report = st.session_state.get("last_report")
    if not summary or not report:
        return

    st.subheader("Run summary")
    top = st.columns(3)
    top[0].metric("SKUs scanned", f"{summary['total_skus']:,}")
    top[1].metric("Flagged for review", f"{summary['flagged_count']:,}")
    top[2].metric("Estimated $ at stake", f"${summary['total_dollar_impact']:,.0f}")

    st.caption("Recommended actions for flagged SKUs")
    actions = st.columns(4)
    counts = summary["action_counts"]
    for column, (action, label) in zip(actions, ACTION_LABELS):
        column.metric(label, f"{counts.get(action, 0):,}")

    st.download_button(
        "Download Excel review",
        data=report,
        file_name="safety_stock_review.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        width="stretch",
    )
    st.caption(
        "The workbook contains a prioritized Review Queue and a full Portfolio Scan. "
        "Start with the Review Queue tab."
    )


st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="wide")

st.title(APP_TITLE)
st.markdown(
    "Upload current planning data, identify safety-stock and reorder-point drift, "
    "and receive a prioritized Excel review queue."
)

with st.expander("What this tool does", expanded=True):
    st.markdown(
        """
1. Loads the item master, weekly demand history, and actual receipt lead times.
2. Recalculates safety stock and reorder point for every SKU using demand variability
   and actual supplier lead-time variability.
3. Flags material drift, trend, intermittency, lead-time drift, and demand outliers.
4. Reviews only flagged SKUs with Claude when an API key is available; otherwise it
   automatically uses the existing transparent rule-based fallback.
5. Produces `safety_stock_review.xlsx` with a prioritized **Review Queue** and a full
   **Portfolio Scan**.

The workbook is decision support. A planner should approve parameter changes through
the normal business process before updating an ERP.
        """
    )

with st.expander("Input rules, file structure, and sample files", expanded=True):
    st.markdown(
        """
- Upload exactly one file for each data set below. CSV files must include a header row.
- Keep SKU values consistent across all three files, including capitalization and
  leading zeros. Every item-master SKU should have demand and receipt history.
- Use one item-master row per SKU, one demand row per SKU/week, and one receipt row per
  received purchase order. Numeric columns must contain numbers only—no `$`, `%`, or
  comma formatting.
- `item_class` should be `A`, `B`, or `C`; lead times are calendar days; demand and stock
  quantities use the item's planning unit; `unit_cost` uses one consistent currency.
- The analyzer uses the most recent 26 demand rows per SKU as the recent window. Sort
  demand history from oldest to newest before uploading.
        """
    )
    st.dataframe(FILE_GUIDE, hide_index=True, width="stretch")

    st.markdown("**Download working sample files**")
    sample_columns = st.columns(3)
    for column, filename in zip(sample_columns, INPUT_FILES):
        sample_path = BASE_DIR / filename
        column.download_button(
            filename,
            data=sample_path.read_bytes(),
            file_name=filename,
            mime="text/csv",
            width="stretch",
        )

st.subheader("1. Choose the data")
source = st.radio(
    "Data source",
    ("Upload my files", "Use bundled sample data"),
    horizontal=True,
    label_visibility="collapsed",
)

uploads = {}
if source == "Upload my files":
    upload_columns = st.columns(3)
    for column, filename in zip(upload_columns, INPUT_FILES):
        uploads[filename] = column.file_uploader(
            filename,
            type=("csv",),
            key=f"upload_{filename}",
            help=f"Select the data set that corresponds to {filename}. It will be staged under this exact name.",
        )
else:
    st.info("The bundled 60-SKU sample data will be used. Your local sample files are not modified.")

st.subheader("2. Choose the reasoning mode")
deployed_key = _deployment_api_key()
session_key = st.text_input(
    "Anthropic API key (optional)",
    type="password",
    help=(
        "Used in memory for this browser session only. It is never written to a file "
        "or included in the progress log. Leave blank to use the administrator key, "
        "if configured, or the rule-based fallback."
    ),
)
if deployed_key:
    st.caption("An administrator API key is configured. A session key entered above takes priority.")
else:
    st.caption("No key is required. Without one, the existing rule-based fallback runs automatically.")

st.subheader("3. Run and download")
run_clicked = st.button(
    "Run analysis",
    type="primary",
    help=CALCULATION_HELP,
)
st.caption("Hover over **Run analysis** to see the calculation and service-level assumptions.")

if run_clicked:
    missing = [filename for filename in INPUT_FILES if source == "Upload my files" and not uploads[filename]]
    if missing:
        st.error("Upload all three required CSV files before running: " + ", ".join(missing))
    else:
        st.session_state.pop("last_report", None)
        st.session_state.pop("last_summary", None)
        status = st.status("Analysis running…", expanded=True)
        progress_bar = st.progress(0, text="Preparing input files")
        log_placeholder = st.empty()
        report_progress = _progress_callback(log_placeholder, progress_bar)

        try:
            with tempfile.TemporaryDirectory(prefix="safety_stock_run_") as temp_dir:
                run_dir = Path(temp_dir)
                _stage_inputs(run_dir, source, uploads)
                result = run_analysis(
                    data_dir=run_dir,
                    api_key=session_key or deployed_key,
                    progress_callback=report_progress,
                )
                _save_run_result(result)
            status.update(label="Analysis complete", state="complete", expanded=False)
            st.success("The Excel review is ready to download.")
        except SafetyStockInputError as exc:
            status.update(label="Input files need attention", state="error", expanded=True)
            st.error(str(exc))
        except Exception as exc:
            status.update(label="Analysis could not complete", state="error", expanded=True)
            st.error(
                "The analysis stopped before a report could be created. "
                f"{type(exc).__name__}: {exc}"
            )

_render_summary()

st.divider()
st.caption(
    "Privacy: uploaded files are copied to an isolated temporary run folder and removed "
    "after the report is created. API keys are not logged or written into the report."
)

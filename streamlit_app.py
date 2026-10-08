#!/usr/bin/env python3
"""Business-user Streamlit interface for the Safety Stock / ROP Drift Agent."""

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from safety_stock_agent import BASE_DIR, SafetyStockInputError, run_analysis
from inventory_assistant import (
    InventoryAssistantError,
    MissingNvidiaAPIKey,
    get_nvidia_config,
    reset_inventory_session,
)
from inventory_contracts import validate_inventory_frames
from inventory_data import InventoryDataModel
from verified_inventory_assistant import (
    PlanningError,
    ask_verified_inventory_assistant,
)


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
                "current_safety_stock, current_rop, unit_cost, supplier"
            ),
        },
        {
            "Exact filename": "demand_history.csv",
            "One row per": "SKU per week",
            "Required columns": "sku, week, demand_qty",
        },
        {
            "Exact filename": "receipt_history.csv",
            "One row per": "Receipt observation (a PO may repeat for partial receipts)",
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


def _nvidia_setting(name: str):
    """Read NVIDIA settings from Streamlit secrets first, then the environment."""
    try:
        value = st.secrets[name]
    except (FileNotFoundError, KeyError):
        value = os.getenv(name)
    return str(value).strip() if value else None


def reset_analysis_state(state=None):
    """Clear analysis and chat together so conversations never cross data sets."""
    state = st.session_state if state is None else state
    reset_inventory_session(state)


def _input_signature(source: str, uploads: dict) -> str:
    digest = hashlib.sha256(source.encode("utf-8"))
    if source == "Use bundled sample data":
        for filename in INPUT_FILES:
            digest.update((BASE_DIR / filename).read_bytes())
    else:
        for filename in INPUT_FILES:
            upload = uploads.get(filename)
            digest.update(filename.encode("utf-8"))
            digest.update(upload.getvalue() if upload else b"<missing>")
    return digest.hexdigest()


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


def _save_run_result(result: dict, input_signature: str):
    st.session_state["last_report"] = Path(result["output_path"]).read_bytes()
    summary = dict(result["analysis_summary"])
    st.session_state["last_summary"] = summary
    st.session_state["analysis_summary"] = summary
    st.session_state["item_master"] = result["item_master"]
    st.session_state["demand_history"] = result["demand_history"]
    st.session_state["receipt_history"] = result["receipt_history"]
    st.session_state["analysis_results"] = result["analysis_results"]
    validated, validation_report = validate_inventory_frames(
        result["item_master"], result["demand_history"], result["receipt_history"]
    )
    st.session_state["assistant_data_model"] = InventoryDataModel.build(
        validated["item_master"],
        validated["demand_history"],
        validated["receipt_history"],
        result["analysis_results"],
    )
    st.session_state["assistant_validation_warnings"] = validation_report.warnings
    st.session_state["review_queue"] = result["review_queue"]
    st.session_state["inventory_chat"] = []
    st.session_state["assistant_metadata"] = {}
    st.session_state["last_input_signature"] = input_signature


def _render_answer_evidence(evidence: dict | None):
    """Show deterministic provenance without prompts, credentials, or model reasoning."""
    if not evidence:
        return
    with st.expander("View data used for this answer"):
        response_paths = evidence.get("response_paths", [])
        if response_paths:
            st.markdown("**Response path:** " + ", ".join(response_paths))
        knowledge_topics = evidence.get("knowledge_topics", [])
        if knowledge_topics:
            st.markdown(
                "**Inventory knowledge topics:** "
                + ", ".join(topic.replace("_", " ") for topic in knowledge_topics)
            )
        sources = evidence.get("source_tables", evidence.get("data_sources_used", []))
        if sources:
            st.markdown("**Sources used:** " + ", ".join(f"`{source}`" for source in sources))
        if evidence.get("result_id"):
            st.caption(
                f"Result: {evidence['result_id']} · Dataset: "
                f"{str(evidence.get('dataset_fingerprint', ''))[:12]}"
            )
        if evidence.get("metric_definition"):
            st.markdown("**Metric definition**")
            st.write(evidence["metric_definition"])
        coverage = evidence.get("coverage", {})
        if coverage:
            st.markdown("**Coverage**")
            st.json(coverage)
        interval = evidence.get("date_interval")
        if interval:
            st.markdown("**Period interval**")
            st.json(interval)
        plan = evidence.get("question_plan", {})
        if plan:
            st.markdown("**Parsed question plan**")
            st.json({
                key: plan.get(key)
                for key in (
                    "query_type", "entity_type", "skus", "suppliers", "metric",
                    "metrics", "direction", "limit", "conditions", "time_window",
                    "confidence",
                )
                if plan.get(key) not in (None, [], {})
            })
        filters = evidence.get("filters", {})
        if filters:
            st.markdown("**Filters**")
            st.json(filters)
        operations = plan.get("operations", [])
        if operations:
            st.markdown("**Python operations:** " + " → ".join(operations))
        definitions = evidence.get("definitions", {})
        if definitions:
            st.markdown("**Metric definitions**")
            st.json(definitions)
        limitations = evidence.get("limitations", []) + evidence.get("warnings", [])
        if limitations:
            st.markdown("**Limitations**")
            for limitation in limitations:
                st.write(f"- {limitation}")
        debug = evidence.get("debug", {})
        if debug:
            st.caption(
                f"Intent: {debug.get('intent', 'unknown')} · "
                f"Rows sent to the explanation layer: {debug.get('rows_sent_to_llm', 0)}"
            )
        source_rows = evidence.get("source_rows", [])
        if source_rows:
            st.markdown("**Source rows**")
            st.dataframe(pd.DataFrame(source_rows), hide_index=True, width="stretch")
        plan_details = evidence.get("plan")
        if plan_details:
            st.markdown("**Validated analytical plan**")
            st.json(plan_details)
        if evidence.get("query"):
            st.markdown("**Executed read-only query**")
            st.code(evidence["query"], language="sql")
        if evidence.get("tie_handling"):
            st.caption(
                f"Tie handling: {evidence['tie_handling']} · "
                f"Calculation complete: {evidence.get('calculation_complete', True)} · "
                f"Display truncated: {evidence.get('display_truncated', False)}"
            )


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


def _render_inventory_assistant():
    st.divider()
    st.subheader("Ask Your Inventory Data")
    if not st.session_state.get("analysis_results"):
        st.info("Run the inventory analysis first to ask questions about your data.")
        return

    api_key = _nvidia_setting("NVIDIA_API_KEY")
    model = _nvidia_setting("NVIDIA_MODEL")
    base_url = _nvidia_setting("NVIDIA_BASE_URL")
    try:
        get_nvidia_config(api_key=api_key, model=model, base_url=base_url)
    except MissingNvidiaAPIKey as exc:
        st.info(str(exc))
        return

    st.caption(
        "NVIDIA Nemotron plans a bounded analytical request; deterministic tools calculate "
        "the values over the current dataset. It can also explain inventory concepts and "
        "offer clearly labeled recommendations from verified context."
    )
    chat = st.session_state.setdefault("inventory_chat", [])
    for message in chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            table = message.get("table")
            if isinstance(table, pd.DataFrame) and not table.empty:
                st.dataframe(table, hide_index=True, width="stretch")
            _render_answer_evidence(message.get("evidence"))

    question = st.chat_input("Ask a question about this completed analysis")
    if not question:
        return
    chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Calculating evidence and asking NVIDIA Nemotron…"):
                config = get_nvidia_config(api_key=api_key, model=model, base_url=base_url)
                from openai import OpenAI
                client = OpenAI(
                    base_url=config["base_url"], api_key=config["api_key"],
                    timeout=30.0, max_retries=1,
                )
                answer = ask_verified_inventory_assistant(
                    question=question,
                    data_model=st.session_state.get("assistant_data_model"),
                    client=client,
                    model=config["model"],
                    prior_context=st.session_state.get("assistant_metadata"),
                    validation_warnings=st.session_state.get("assistant_validation_warnings", []),
                )
            st.markdown(answer.text)
            if answer.table is not None and not answer.table.empty:
                st.dataframe(answer.table, hide_index=True, width="stretch")
            _render_answer_evidence(answer.evidence)
            chat.append({
                "role": "assistant",
                "content": answer.text,
                "table": answer.table,
                "evidence": answer.evidence,
            })
            st.session_state["assistant_metadata"] = answer.metadata
            st.session_state["inventory_chat"] = chat[-12:]
        except (InventoryAssistantError, PlanningError) as exc:
            st.error(str(exc))
            chat.append({"role": "assistant", "content": str(exc)})
            st.session_state["inventory_chat"] = chat[-12:]
        except (ValueError, KeyError) as exc:
            message = f"I could not analyze that question: {exc}"
            st.error(message)
            chat.append({"role": "assistant", "content": message})
            st.session_state["inventory_chat"] = chat[-12:]


st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="wide")

st.title(APP_TITLE)
st.markdown(
    "Upload current planning data, identify safety-stock and reorder-point drift, "
    "and receive a prioritized Excel review queue."
)

with st.expander("What this tool does", expanded=True):
    st.markdown(
        "**Formula and service-level assumptions**",
        help=CALCULATION_HELP,
    )
    st.markdown(
        """
1. Loads the item master, weekly demand history, and actual receipt lead times.
2. Recalculates safety stock and reorder point for every SKU using demand variability
   and actual supplier lead-time variability.
3. Flags material drift, trend, intermittency, lead-time drift, and demand outliers.
4. Reviews only flagged SKUs with Claude when an administrator API key is configured;
   otherwise it automatically uses the existing transparent rule-based fallback.
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
- Use one item-master row per SKU, one demand row per SKU/week, and one row per
  receipt observation. A PO may repeat for partial receipts; it is not silently
  collapsed to a PO-level statistic. Numeric columns must contain numbers only—no `$`, `%`, or
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

current_input_signature = _input_signature(source, uploads)
previous_input_signature = st.session_state.get("last_input_signature")
if previous_input_signature and current_input_signature != previous_input_signature:
    reset_analysis_state()
    st.info("The selected input data changed. Run the analysis to refresh results and start a new chat.")

deployed_key = _deployment_api_key()
st.subheader("2. Run and download")
run_clicked = st.button("Run analysis", type="primary")

if run_clicked:
    missing = [filename for filename in INPUT_FILES if source == "Upload my files" and not uploads[filename]]
    if missing:
        st.error("Upload all three required CSV files before running: " + ", ".join(missing))
    else:
        reset_analysis_state()
        transient_output = st.empty()
        with transient_output.container():
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
                    api_key=deployed_key,
                    progress_callback=report_progress,
                )
                _save_run_result(result, current_input_signature)
            transient_output.empty()
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
_render_inventory_assistant()

st.divider()
st.caption(
    "Privacy: uploaded files are copied to an isolated temporary run folder and removed "
    "after the report is created. API keys are not logged or written into the report."
)

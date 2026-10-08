# Inventory assistant architecture

## Current responsibilities and observed bottlenecks

- `streamlit_app.py` stages the three CSV uploads, runs the SS/ROP workflow, stores
  session-local data, and renders chat. Upload content changes already clear the
  prior chat state.
- `safety_stock_agent.py` validates uploads, runs the authoritative SS/ROP
  formula, optionally asks Claude to review flagged recommendations, and writes
  the workbook.
- `inventory_data.py` independently aggregates demand and receipt facts before
  joining either to the one-row-per-SKU item dimension. This avoids the
  demand-by-receipt multiplication caused by a raw many-to-many join.
- `inventory_assistant.py` contains provider configuration, a large set of
  pandas evidence handlers, response prompting, and compatibility entry points.
- `inventory_question_parser.py` is a phrase/keyword interpreter. It recognizes
  several examples but is brittle and duplicates analytical routing logic.

The main bottlenecks are the keyword interpreter being the primary planner,
coverage metadata being absent from most results, provenance not having a typed
contract, and PO source rows not being retained in ranking results. The
OpenAI-compatible NVIDIA endpoint is configured in code. Native tool calling or
provider-enforced structured output is not established by the repository, so
the verified assistant uses a validated JSON planning protocol with one repair
attempt. Provider code remains separate from execution.

## Canonical source mappings

| Table | Grain | Required columns |
| --- | --- | --- |
| `item_master` | one row per SKU | `sku`, `description`, `supplier`, `item_class`, `assumed_lead_time_days`, `current_safety_stock`, `current_rop`, `unit_cost` |
| `demand_history` | one row per SKU/week | `sku`, `week`, `demand_qty` |
| `receipt_history` | one receipt observation | `sku`, `po_number`, `actual_lead_time_days` |

`supplier` is the current primary-supplier mapping. Receipt rows do not contain
a supplier field or dates. `po_number` is present, but repeated SKU/PO values may
represent partial receipts and are therefore not silently collapsed into a
single PO statistic. The source does not include promised date, receipt date,
quantity, cost by PO, OTIF, fill rate, or quality. Those questions are reported
as unsupported. ABC class is uploaded in `item_master`; the assistant does not
reclassify it.

## Metric conventions

- Lead time is in calendar days. Variability is the sample standard deviation
  (`ddof=1`). At least two receipt observations are required for a SKU to be
  eligible for variability ranking; one observation is reported as insufficient.
- Demand variability is sample standard deviation; coefficient of variation is
  undefined when mean demand is zero.
- SS/ROP calculations remain in `safety_stock_agent.analyze_sku`: the latest 26
  demand rows are converted from weekly to daily values, combined demand and
  lead-time variance is used, and input ABC classes map to z-values A=2.05,
  B=1.65, C=1.28 (unknown values retain the existing B default).
- “Last six weeks” is dataset-relative because the sample `week` field is an
  ordinal. Evidence shows the exact first and last period values used. Growth is
  recent-six mean versus the immediately preceding six-period mean. “Increasing”
  means positive growth unless the user supplies another threshold. “Highly
  inconsistent” has no business threshold and requires clarification unless it
  is used only as ranking language.

## Target flow

1. Normalize and validate the uploaded frames and compute a dataset fingerprint.
2. Give the planning model the exact data dictionary, allowed operations,
   response paths, and bounded structured conversation context—not raw source rows.
3. Validate the returned JSON plan. It may select analysis, general inventory
   knowledge, contextual recommendation, or a mixture. Analysis executes at most
   three approved read-only operations.
4. Use DuckDB only through the structural SQL validator; external access and
   extension loading are disabled and execution is interruptible.
5. Return a typed evidence result with executor-computed coverage, identifiers,
   units, source rows, warnings, and truncation state.
6. Render critical facts and tables deterministically. General knowledge needs no
   query. Recommendation generation receives bounded verified evidence and must
   label causes as hypotheses and actions as proposals; unverified identifiers or
   numbers trigger a curated fallback.
7. Conversation context is updated only after successful validated execution and
   is scoped to the dataset fingerprint and Streamlit session. Conceptual turns
   preserve rather than overwrite the current verified SKU/PO context.

`inventory_question_parser.py` remains only as a compatibility and validation
surface for older direct callers/tests; it is not the primary path used by the
verified assistant.

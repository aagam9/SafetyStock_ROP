# Safety Stock / ROP Drift Review

A Streamlit inventory-planning application that recalculates Safety Stock and
Reorder Point (ROP), produces an Excel review queue, and answers natural-language
questions through a validated analytical interface.

The existing calculation engine remains authoritative. NVIDIA Nemotron plans an
allow-listed request; Python/DuckDB calculates the answer over the current data.
Critical identifiers and values are rendered from typed evidence rather than
model prose.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The command-line workbook workflow remains available:

```bash
python safety_stock_agent.py
```

In the web app, upload the three CSVs (or choose the bundled sample), run the
analysis, review/download the workbook, then use **Ask Your Inventory Data** if
NVIDIA is configured. A changed upload or new run clears chat and structured
references so evidence cannot cross datasets.

## Input contracts

Headers are case-sensitive. SKU values must be consistent, including leading
zeros and capitalization.

### `item_master.csv`

Grain: one row per SKU.

| Column | Meaning |
| --- | --- |
| `sku` | Nonblank unique item identifier |
| `description` | Item description |
| `supplier` | Current primary supplier mapping |
| `item_class` | Uploaded ABC class (`A`, `B`, or `C`) |
| `assumed_lead_time_days` | Current planning lead time in calendar days |
| `current_safety_stock` | Current Safety Stock quantity |
| `current_rop` | Current ROP quantity |
| `unit_cost` | Positive unit cost in one consistent currency |

The app does not recompute ABC class and does not assume multi-sourcing.

### `demand_history.csv`

Grain: one row per SKU/week or period.

| Column | Meaning |
| --- | --- |
| `sku` | Item identifier in `item_master.csv` |
| `week` | Consistently numeric period or parseable date, unique within SKU |
| `demand_qty` | Nonnegative period demand |

The SS/ROP calculation uses the most recent 26 rows. Analytical windows sort a
numeric/date `week` field and expose the actual period interval used.

### `receipt_history.csv`

Grain: one receipt observation. A PO may have multiple rows for partial receipts;
the source has no receipt-line identifier or quantity.

| Column | Meaning |
| --- | --- |
| `sku` | Item identifier in `item_master.csv` |
| `po_number` | Nonblank source PO identifier; may repeat |
| `actual_lead_time_days` | Positive observed lead time in calendar days |

Repeated SKU/PO rows are retained and disclosed. Exact duplicate
SKU/PO/lead-time rows are rejected because the current fields cannot distinguish
them from accidental duplication. The app does not invent a PO-level lead-time
aggregation convention.

Validation also rejects missing fields, blank identifiers, duplicate item SKUs,
duplicate SKU/week rows, invalid numeric values, inconsistent period formats,
negative demand, nonpositive lead times, unknown SKUs, and missing histories.

## Safety Stock and ROP calculations

`safety_stock_agent.analyze_sku` remains the authoritative engine. It uses:

- the latest 26 demand rows, converted from weekly to daily values;
- sample demand and lead-time standard deviations (`ddof=1`);
- receipt lead times in calendar days;
- combined demand and lead-time variance;
- uploaded ABC classes with z-values A=2.05, B=1.65, and C=1.28;
- `ROP = mean lead-time demand + Safety Stock`.

The formulas, drift thresholds, trend/outlier rules, and workbook fields are in
`CALCULATION_METHODOLOGY.txt`. Without `ANTHROPIC_API_KEY`, the workbook's
existing transparent rule-based review still runs.

## NVIDIA setup and capability decision

The assistant uses the existing OpenAI-compatible NVIDIA endpoint. Secrets are
not logged or written to reports.

```text
NVIDIA_API_KEY=your_api_key_here
NVIDIA_MODEL=nvidia/nemotron-3-super-120b-a12b
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

Only `NVIDIA_API_KEY` is required. Environment variables and Streamlit secrets
take precedence over a local ignored `.env` file. Do not commit `.env`,
`settings.json`, or `.streamlit/secrets.toml`.

The repository does not establish reliable native tool calling or
provider-enforced structured outputs for the configured endpoint. The assistant
therefore uses JSON planning with schema validation and one repair attempt. It
sends the schema, data dictionary, entity catalog, and bounded structured
conversation context—not bulk source rows—to the planning model.

## Verified question flow

1. Validate/normalize source frames and compute a dataset fingerprint.
2. Ask the model for a semantic JSON plan selecting one or more response paths:
   `analysis`, `knowledge`, and `recommendation`.
3. Validate the outcome, response paths, operations, arguments, entity references,
   knowledge topics, and SQL.
4. For analysis, execute deterministic calculations over the complete applicable
   population and render identifiers/numbers from typed evidence.
5. For general knowledge, explain definitions, methodology, and concepts without
   requiring an analytical query.
6. For contextual recommendations, combine bounded verified-result context with
   inventory knowledge. Possible causes are labeled as hypotheses and proposed
   actions are kept separate from uploaded-data findings.
7. Mixed questions can execute analysis and add conceptual explanation or actions
   in the same response. Context updates only from successful tool results.

Reusable operations cover metric ranking, exact SKU comparison, dataset-relative
demand growth, longest source receipt/PO lookup, and repetition of prior verified
results. Flexible questions may use one structurally validated DuckDB SELECT/CTE.
SQL AST checks reject multiple statements, writes, pragmas, attachments,
extensions, external/unapproved tables and functions, and SQL `LIMIT`. DuckDB
external access is disabled, memory/row limits apply, and execution has an
interrupt deadline. Display truncation occurs only after the full query result is
calculated.

Rankings evaluate the complete eligible population before applying the requested
display count, return cutoff ties, and preserve full precision for ordering.
Lead-time variability is sample standard deviation and requires at least two
receipt observations. One observation is reported as insufficient rather than
as zero variability.

“Last six weeks” is dataset-relative unless real dates are uploaded. Evidence
shows the reference and period bounds. Growth means the recent-window mean versus
the immediately preceding equal window. “Increasing” defaults to positive growth
and is disclosed. “Highly inconsistent” needs a business threshold when used as
a filter; ranking superlatives do not.

The structured context retains current SKU/PO sets, metric, filters, interval,
original questions, and up to three prior result references. Ambiguous references
produce clarification. The sequence “most variable SKU” → “longest PO for this
SKU” → “specific PO please” resolves through verified result IDs and source rows.
After a verified MAT-1019 variability result, “How can we reduce this variability?”
uses that SKU and evidence without asking the user to repeat it.

Conceptual prose is separately validated. Recommendation text cannot introduce a
SKU, PO, or numerical claim absent from verified evidence. If generation is
unavailable or violates that boundary, the app uses a curated methodology answer
instead of blocking the conceptual response or weakening numerical grounding.

## Evidence UI

The expandable evidence section shows the result/dataset IDs, source tables,
metric definition and units, executor-computed coverage/exclusions, filters,
period interval, source rows, validated plan, executed query when applicable,
tie policy, calculation completeness, and display truncation.

## Known limitations

- `supplier` is the current primary mapping, not historical PO supplier.
- Receipt history has no order/receipt timestamp, promised date, receipt quantity,
  OTIF, fill rate, quality, or PO cost. Those analyses are unsupported.
- The input has no inventory-position history, backorders, or historical stockout
  events. Stockout Risk is a relative planning score, not a stockout probability.
- Historical demand analysis is not presented as a validated forecast. Forecast
  requests are unsupported by the verified assistant.
- A repeated PO can identify source receipt rows, but the app does not claim a
  PO-level aggregate without a business convention.
- Planner approval is still required before changing ERP parameters.
- Mock-provider success does not prove that every live-model phrasing succeeds.

## File map

- `inventory_contracts.py`: canonical schemas, validation, data dictionary, and
  dataset fingerprinting.
- `inventory_data.py`: independent fact aggregation and SKU relationship view.
- `inventory_analytics.py`: typed evidence, deterministic tools, SQL policy,
  DuckDB registration, limits, and deadlines.
- `verified_inventory_assistant.py`: provider JSON planning, execution,
  follow-up context, and deterministic rendering.
- `inventory_question_parser.py`: compatibility parser/validator for older direct
  callers; it is not the verified chat path's primary interpreter.
- `streamlit_app.py`: upload/run flow, session isolation, chat, and evidence UI.
- `safety_stock_agent.py`: authoritative SS/ROP engine and workbook output.

See `docs/ARCHITECTURE.md` for the architecture note and metric conventions.

## Tests

The deterministic and scripted-provider suites require no live API key:

```bash
python -m unittest discover -s tests -v
```

The current suite contains 78 tests covering contracts and grain, partial receipts and exact duplicates,
many-to-many prevention, sample sizes, direct expected statistics, global
ranking coverage/ties, PO source retrieval, dataset-relative windows, SQL
restrictions, untrusted CSV text, malformed plans, provider outages, dataset
invalidation, session isolation/follow-ups, semantic response-path planning,
general definitions, contextual recommendations, mixed questions, conceptual
grounding fallback, and existing SS/ROP regressions.

Optional live-model evaluation requires `NVIDIA_API_KEY` and must report the
model configuration, repeated-run results, planning errors, and grounding
failures separately from deterministic tool results.

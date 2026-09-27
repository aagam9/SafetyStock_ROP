# Safety Stock / ROP Drift Review

A Streamlit inventory-planning application that recalculates Safety Stock and
Reorder Point (ROP), produces an Excel review queue, and enables grounded
questions about the completed analysis through NVIDIA Nemotron.

The existing calculation engine remains authoritative. Python performs all
filtering, joins, aggregation, forecasting, gap ranking, and Stockout Risk
scoring. Nemotron receives a small question-specific evidence packet and
explains the results; it is not used as a database or calculation engine.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The command-line analysis remains available:

```bash
python safety_stock_agent.py
```

The web workflow is:

1. Upload the three CSV files or choose the bundled sample data.
2. Select **Run analysis**.
3. Review the run summary and download `safety_stock_review.xlsx`.
4. If NVIDIA is configured, use **Ask Your Inventory Data** for grounded
   questions and follow-ups.

New uploads or a newly run analysis clear the prior assistant conversation so
results from different data sets cannot be mixed.

## Input files

CSV headers are case-sensitive. Keep SKU identifiers consistent, including
capitalization and leading zeros.

### `item_master.csv`

One row per SKU.

| Column | Meaning |
| --- | --- |
| `sku` | Nonblank unique SKU identifier |
| `description` | Item description |
| `supplier` | Primary/current supplier for this SKU |
| `item_class` | ABC class (`A`, `B`, or `C`) |
| `assumed_lead_time_days` | Current planning lead time in calendar days |
| `current_safety_stock` | Current Safety Stock setting |
| `current_rop` | Current ROP setting |
| `unit_cost` | Unit cost in one consistent currency |

This version assumes one primary supplier per SKU. It does not implement
multi-sourcing.

### `demand_history.csv`

One row per SKU/week.

| Column | Meaning |
| --- | --- |
| `sku` | Item identifier present in `item_master.csv` |
| `week` | Week number or valid date; unique within the SKU |
| `demand_qty` | Nonnegative demand quantity |

Keep rows in oldest-to-newest order within each SKU. The calculation engine
uses the most recent 26 rows as its recent-demand window.

### `receipt_history.csv`

One row per received purchase order.

| Column | Meaning |
| --- | --- |
| `sku` | Item identifier present in `item_master.csv` |
| `po_number` | Nonblank unique receipt/PO identifier |
| `actual_lead_time_days` | Positive actual receipt lead time in calendar days |

Supplier is joined from `item_master.csv`; it is not required in receipt
history for this single-sourcing version.

Validation rejects missing columns, blank SKU/supplier values, duplicate item
SKUs, duplicate SKU/week rows, duplicate PO numbers, invalid numeric values,
malformed week values, negative demand, nonpositive lead time, unknown SKUs,
and item-master SKUs without demand or receipt history.

## Existing Safety Stock and ROP analysis

For each SKU, the engine calculates recent weekly demand mean and variability,
demand trend, intermittency, demand outliers, actual lead-time mean and
variability, recommended Safety Stock, and recommended ROP. ABC classes select
the existing service-level Z-scores. The formulas and thresholds are documented
in `CALCULATION_METHODOLOGY.txt`.

Only flagged SKUs enter the review queue. The legacy review step can optionally
use Claude through `ANTHROPIC_API_KEY`; without that key, the existing
transparent rule-based review still runs. This is separate from the new
Nemotron conversational assistant.

The workbook contains:

- **Review Queue**: flagged SKUs, supplier, current/recommended settings,
  drift, dollar impact, action, confidence, risk, and rationale.
- **Portfolio Scan**: every SKU with supplier, demand/lead-time statistics,
  current/recommended Safety Stock and ROP, and review flags.

## NVIDIA Nemotron setup

The assistant calls NVIDIA's OpenAI-compatible hosted endpoint with the
`openai` Python client. No credential is stored in source or generated reports.

Required:

```text
NVIDIA_API_KEY=your_api_key_here
```

Optional overrides:

```text
NVIDIA_MODEL=nvidia/nemotron-3-super-120b-a12b
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

The model and endpoint are centralized in `inventory_assistant.py`. Set the
variables in the environment before starting Streamlit. `.env.example` is a
non-secret template; the application does not automatically load `.env` files.

For Streamlit deployment, the equivalent `.streamlit/secrets.toml` is:

```toml
NVIDIA_API_KEY = "your-key-here"
NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
```

Obtain an API key from the NVIDIA API catalog/account used by your
organization. Do not commit `.env`, `settings.json`, or
`.streamlit/secrets.toml`; these paths are ignored by Git.

If `NVIDIA_API_KEY` is absent or the hosted service fails, the inventory
analysis and Excel report continue to work. Only the assistant is unavailable.

## Assistant capabilities

The grounded assistant supports:

- SKU demand behavior, trends, variability, and intermittency
- deterministic weekly demand forecasts for a named SKU
- current versus recommended Safety Stock and ROP
- Safety Stock and ROP gap rankings
- an explainable relative **Stockout Risk** ranking
- receipt lead-time averages and variability
- unusually long/inconsistent receipt behavior
- supplier lead-time performance and consistency
- supplier and SKU comparisons
- bounded conversational follow-ups

Forecasts use up to the latest 13 observations, fit a simple linear trend, and
damp that trend by 50%. They are transparent planning estimates rather than
promises of future demand.

Stockout Risk is a relative 0-100 planning score based on Safety Stock gap, ROP
gap, demand variability, lead-time variability, positive trend, and ABC
criticality. It is not a probability of stockout.

## Data and analytical limitations

- The inputs do not contain inventory-position history, backorders, or explicit
  stockout events. The app cannot report historical stockout counts.
- Supplier analysis is currently lead-time performance and consistency, not a
  complete supplier scorecard.
- OTIF, fill rate, quality, and promised-date adherence require fields that are
  not present in the current schemas.
- Forecasts are estimates based only on uploaded demand history; they do not
  include promotions, pricing, capacity, or external drivers.
- Nemotron explains deterministic evidence. It does not replace the Safety
  Stock/ROP engine or independently calculate portfolio metrics.
- Planner approval is required before changing ERP planning parameters.

## Tests

The test suite uses no real NVIDIA calls and requires no API key:

```bash
python -m unittest discover -s tests -v
```

Coverage includes supplier validation/joining, SKU retrieval, supplier
aggregation, deterministic forecasts, Safety Stock/ROP rankings, Stockout Risk,
unsupported-data responses, context construction, session reset, missing NVIDIA
configuration, and a mocked Nemotron call.

# Safety Stock / ROP Drift Agent

Detects safety stock and reorder point parameters that have drifted from what
they statistically should be — then uses Claude to make the judgment call on
ambiguous cases (trends, seasonality, outliers, thin data) that pure math
can't resolve on its own.

## Why this exists
Safety stock/ROP is usually set once, from whatever demand and lead-time
assumptions existed at the time, and then never revisited — until a stockout
or an inventory audit forces the issue. This agent proactively finds the SKUs
where that gap has become real, and tells you *why*, not just *that*.

## How it works
1. **Statistics do the screening.** For every SKU, it recalculates safety
   stock/ROP using actual recent demand variability and actual supplier lead
   time performance (the combined demand + lead-time variance formula), and
   compares it to what's currently set.
2. **Only real drift gets escalated.** SKUs where the numbers still hold up
   are auto-cleared — no need to review them.
3. **The agent makes the judgment call.** For flagged SKUs, Claude reviews the
   drift alongside trend/seasonality/outlier signals and decides: increase,
   decrease, hold, or flag for manual review (e.g. because an outlier order is
   probably distorting the recommendation) — with a plain-language rationale.
4. **Output:** `safety_stock_review.xlsx` — a prioritized Review Queue tab
   (ranked by $ impact, with anything needing manual review pushed to the top
   regardless of size) and a Portfolio Scan tab covering every SKU.

## Files
- `generate_data.py` — creates the synthetic sample data (item master, 2
  years of weekly demand, PO receipt history). Run once to get started.
- `safety_stock_agent.py` - the agent itself. Reads the CSVs, runs the
  analysis, calls Claude for flagged SKUs, writes the Excel output.
- `streamlit_app.py` - the business-user web interface.
- `CALCULATION_METHODOLOGY.txt` - calculation formulas, thresholds,
  assumptions, fallback rules, limitations, and governance review checklist.
- `requirements.txt` and `.streamlit/config.toml` - deployment dependencies
  and presentation settings.
- `item_master.csv`, `demand_history.csv`, `receipt_history.csv` - sample data
  already generated for you (60 SKUs).
- `safety_stock_review.xlsx` - sample output from a run.

## Run the web app locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`. Open it in a
browser, upload the three CSVs (or choose the bundled sample data), and select
**Run analysis**. The completed Excel workbook is downloaded from the page.

Uploads are staged in a separate temporary directory for each run and removed
after the workbook is captured. API keys are not entered through the planner
interface, logged, or written to the report.

## Keep using the command line

The original command remains available:

```bash
python safety_stock_agent.py
```

It reads the three CSVs beside the script and writes
`safety_stock_review.xlsx` there, just as before. To regenerate the bundled
sample data, run:

```bash
python generate_data.py
```

## File naming and structure

Use these exact data sets. The web app has one clearly labeled upload slot for
each file and stages it under the correct name.

| Filename | One row per | Columns |
| --- | --- | --- |
| `item_master.csv` | SKU | `sku`, `description`, `item_class`, `assumed_lead_time_days`, `current_safety_stock`, `current_rop`, `unit_cost` |
| `demand_history.csv` | SKU/week | `sku`, `week`, `demand_qty` |
| `receipt_history.csv` | Received PO | `sku`, `po_number`, `actual_lead_time_days` |

Business data rules:

- Include a header row and keep SKU values consistent across all files,
  including capitalization and leading zeros.
- Use one item-master row per SKU, one demand row per SKU/week, and one receipt
  row per received PO.
- Sort each SKU's demand rows from oldest to newest. The last 26 rows are used
  as the recent demand window.
- Keep numeric columns numeric: do not include currency signs, percent signs,
  or thousands separators.
- Use `A`, `B`, or `C` for `item_class`; calendar days for lead time; the same
  planning unit for demand and inventory; and one consistent unit-cost currency.

The three CSVs already in this folder are working samples. They can also be
downloaded directly from the web app.

## Deploy on Streamlit Community Cloud

The project is laid out for Streamlit Community Cloud:

1. Put this folder in a GitHub repository. Do not commit `settings.json` or
   `.streamlit/secrets.toml`; both are excluded by `.gitignore`.
2. Sign in at `share.streamlit.io`, choose **Create app**, and select the
   repository, branch, and `streamlit_app.py` as the entry point.
3. In **Advanced settings**, select Python 3.12.
4. Optional: add a shared administrator key in the Secrets field:

   ```toml
   ANTHROPIC_API_KEY = "your-key-here"
   ```

5. Deploy and share the generated `streamlit.app` URL with planners.

If an administrator key is configured, all users can invoke Claude against
that account. Restrict app access and monitor usage/costs accordingly. If no
shared key is configured, the web app automatically uses the rule-based
fallback.

Actual cloud publication requires access to the target GitHub repository and
Streamlit workspace. The code and dependency/configuration files in this
folder are ready for that final account-linked step.

## API key options

For the CLI, either set an environment variable:
```bash
export ANTHROPIC_API_KEY="your_key_here"
```
or create a `settings.json` next to the script:
```json
{"ANTHROPIC_API_KEY": "your_key_here"}
```
Without a key, it runs with a transparent rule-based fallback so the pipeline
still works end-to-end — you just lose the nuanced judgment on ambiguous
cases (that's the whole point of the agent, so add the key when you can).

Most ERPs (SAP MD04/MC.9, Oracle, etc.) can export exactly this via standard
reports - material master, historical consumption, and PO history with actual
vs. planned dates.

## Tuning
In `safety_stock_agent.py`:
- `DRIFT_THRESHOLD_PCT` — how much statistical drift before a SKU is flagged
- `LT_DRIFT_THRESHOLD` — how much lead-time drift before flagging
- `CLASS_Z` — service-level Z-scores per item class (A/B/C)
- `RECENT_WINDOW_WEEKS` — how far back "recent" demand looks for trend
  comparison

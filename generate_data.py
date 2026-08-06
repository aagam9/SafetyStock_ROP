#!/usr/bin/env python3
"""
Generates synthetic ERP-style exports for the Safety Stock / ROP Drift agent:
  - item_master.csv     : SKU, class, current safety stock/ROP, assumed lead time, cost
  - demand_history.csv  : 104 weeks of actual weekly demand per SKU
  - receipt_history.csv : actual PO receipt lead times per SKU (last ~15-25 POs)

This mimics what you'd export from SAP/Oracle: item master, historical
consumption, and PO history with actual vs. planned lead times.
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent
RNG = np.random.default_rng(42)

N_WEEKS = 104          # 2 years of weekly demand history
CLASS_Z = {"A": 2.05, "B": 1.65, "C": 1.28}   # service level Z-scores (98% / 95% / 90%)
CLASS_WEIGHTS = {"A": 0.2, "B": 0.5, "C": 0.3}

# Demand pattern archetypes — each SKU is assigned one
PATTERNS = [
    "stable", "stable", "stable",          # steady demand, current params probably fine
    "growing", "declining",                # trend — current params likely stale
    "seasonal",                            # cyclical — flat-average params understate peaks
    "intermittent",                        # lumpy/sparse — high variance, easy to mis-set
    "spike_distorted",                     # one-time bulk order skews the history
]

LT_DRIFT_TYPES = [
    "stable", "stable",        # actual LT matches assumed LT
    "slower",                  # supplier has quietly gotten slower
    "faster",                  # supplier improved, current SS may be excessive
]


def gen_demand(pattern: str, base: float) -> np.ndarray:
    weeks = np.arange(N_WEEKS)
    noise_scale = base * 0.25
    if pattern == "stable":
        d = base + RNG.normal(0, noise_scale, N_WEEKS)
    elif pattern == "growing":
        d = base * (1 + 0.006 * weeks) + RNG.normal(0, noise_scale, N_WEEKS)
    elif pattern == "declining":
        d = base * (1 - 0.006 * weeks).clip(min=0.15) + RNG.normal(0, noise_scale, N_WEEKS)
    elif pattern == "seasonal":
        d = base * (1 + 0.5 * np.sin(2 * np.pi * weeks / 52)) + RNG.normal(0, noise_scale, N_WEEKS)
    elif pattern == "intermittent":
        occurs = RNG.random(N_WEEKS) < 0.35
        d = np.where(occurs, RNG.normal(base * 3, base, N_WEEKS), 0)
    elif pattern == "spike_distorted":
        d = base + RNG.normal(0, noise_scale, N_WEEKS)
        # one enormous one-off order in the last 6 months
        spike_week = RNG.integers(N_WEEKS - 26, N_WEEKS)
        d[spike_week] += base * 12
    else:
        d = base + RNG.normal(0, noise_scale, N_WEEKS)
    return np.clip(d, 0, None).round(0)


def gen_lead_times(assumed_lt: int, drift: str, n_pos: int) -> np.ndarray:
    """Chronological PO lead times (index 0 = oldest, -1 = most recent). For
    'slower'/'faster', the drift emerges gradually over the series rather than
    being present from day one — mirroring a supplier that has changed over time."""
    idx = np.arange(n_pos)
    progress = idx / max(n_pos - 1, 1)   # 0 -> 1 across the PO history
    if drift == "stable":
        target = np.full(n_pos, assumed_lt)
    elif drift == "slower":
        target = assumed_lt * (1 + 0.55 * progress)
    elif drift == "faster":
        target = assumed_lt * (1 - 0.35 * progress)
    else:
        target = np.full(n_pos, assumed_lt)
    lt = RNG.normal(target, assumed_lt * 0.10)
    return np.clip(lt, 1, None).round(1)


def main():
    n_sku = 60
    item_rows, demand_rows, receipt_rows = [], [], []

    categories = ["Fastener", "Bearing", "Gasket", "Circuit Board", "Motor",
                  "Sensor", "Casting", "Valve", "Wiring Harness", "Bracket"]

    for i in range(1, n_sku + 1):
        sku = f"MAT-{1000 + i}"
        item_class = RNG.choice(list(CLASS_WEIGHTS.keys()), p=list(CLASS_WEIGHTS.values()))
        pattern = RNG.choice(PATTERNS)
        lt_drift = RNG.choice(LT_DRIFT_TYPES)
        assumed_lt_days = int(RNG.choice([21, 30, 45, 60, 75]))
        base_weekly_demand = float(RNG.uniform(15, 400))
        unit_cost = float(round(RNG.uniform(2, 250), 2))

        demand = gen_demand(pattern, base_weekly_demand)
        for w, qty in enumerate(demand):
            demand_rows.append({"sku": sku, "week": w + 1, "demand_qty": qty})

        n_pos = int(RNG.integers(12, 26))
        lts = gen_lead_times(assumed_lt_days, lt_drift, n_pos)
        for j, lt in enumerate(lts):
            receipt_rows.append({"sku": sku, "po_number": f"PO-{sku}-{j+1:03d}",
                                  "actual_lead_time_days": lt})

        # Current SS/ROP as if set once, long ago: computed with the SAME
        # combined demand+lead-time-variance formula the agent uses today, but
        # based only on the EARLY portion of demand history and the EARLIEST
        # POs on file -- i.e. whatever was knowable at the time the parameter
        # was set and never revisited. For SKUs whose demand pattern and
        # supplier lead time haven't really changed since, this will land close
        # to today's recalculation. For SKUs that have genuinely shifted
        # (trend, seasonality, a supplier that's drifted), it won't -- which is
        # the real-world "set it and forget it" problem this agent looks for.
        early_window = demand[:min(26, len(demand))]
        early_lts = lts[:max(len(lts) // 2, 4)]
        z = CLASS_Z[item_class]
        sigma_d_initial_daily = (early_window.std(ddof=1) if len(early_window) > 1 else 0.0) / 7
        mean_d_initial_daily = early_window.mean() / 7
        lt_initial_mean = float(early_lts.mean())
        lt_initial_std = float(early_lts.std(ddof=1)) if len(early_lts) > 1 else 0.0
        variance_term = (lt_initial_mean * sigma_d_initial_daily ** 2
                          + mean_d_initial_daily ** 2 * lt_initial_std ** 2)
        current_ss = round(z * np.sqrt(max(variance_term, 0)))
        current_rop = round(mean_d_initial_daily * lt_initial_mean + current_ss)

        item_rows.append({
            "sku": sku,
            "description": f"{RNG.choice(categories)} - {sku}",
            "item_class": item_class,
            "assumed_lead_time_days": assumed_lt_days,
            "current_safety_stock": int(current_ss),
            "current_rop": int(current_rop),
            "unit_cost": unit_cost,
        })

    pd.DataFrame(item_rows).to_csv(OUT_DIR / "item_master.csv", index=False)
    pd.DataFrame(demand_rows).to_csv(OUT_DIR / "demand_history.csv", index=False)
    pd.DataFrame(receipt_rows).to_csv(OUT_DIR / "receipt_history.csv", index=False)

    print(f"Generated {n_sku} SKUs:")
    print(f"  item_master.csv     ({len(item_rows)} rows)")
    print(f"  demand_history.csv  ({len(demand_rows)} rows)")
    print(f"  receipt_history.csv ({len(receipt_rows)} rows)")


if __name__ == "__main__":
    main()

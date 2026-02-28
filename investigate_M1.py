#!/usr/bin/env python3
"""M1: Triplet STDP mechanism investigation.

Investigates the role of pair-based and triplet-based STDP in OSI emergence.

Conditions (12 total):
  1. STDP OFF — all plasticity amplitudes zeroed
  2. Pair-only — triplet terms zeroed (A3_plus=0, A3_minus=0)
  3. LTP/LTD ratio sweep — A2_minus in {0.004, 0.006, 0.008, 0.010, 0.014, 0.020}
  4. Triplet boost sweep — A3_plus in {0.001, 0.002, 0.004}
     (A3_plus=0.0 already covered by pair-only condition)

Level-3 deep analysis: bimodality index, weak/strong weight fractions at each checkpoint.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from osi_investigation_harness import (
    InvestigationConfig,
    run_investigation,
    save_condition_result,
    plot_osi_timeseries,
    plot_dose_response,
    plot_pref_trajectories,
    plot_weight_evolution,
    write_condition_report,
)

import numpy as np
from scipy import stats

OUT = "investigation_results/M1_triplet_stdp"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Mechanism-specific checkpoint diagnostics
# ---------------------------------------------------------------------------

def m1_checkpoint_fn(net, segment):
    """Compute STDP-specific diagnostics at each checkpoint.

    Metrics per ensemble:
      - bimodal_coeff : Sarle's bimodal coefficient = (skewness^2 + 1) / kurtosis
      - weak_frac     : fraction of weights < 0.1 * w_max
      - strong_frac   : fraction of weights > 0.9 * w_max
    """
    W = net.W.copy()
    M = net.M
    w_max = net.p.w_max

    bimodal = []
    weak_frac = []
    strong_frac = []

    for m in range(M):
        w = W[m].ravel()
        sk = float(stats.skew(w))
        ku = float(stats.kurtosis(w, fisher=False))  # Pearson kurtosis (non-excess)
        bc = (sk**2 + 1) / ku if ku > 0 else 0.0
        bimodal.append(bc)
        weak_frac.append(float((w < 0.1 * w_max).mean()))
        strong_frac.append(float((w > 0.9 * w_max).mean()))

    return {
        "bimodal_coeff": bimodal,
        "weak_frac": weak_frac,
        "strong_frac": strong_frac,
    }


# ---------------------------------------------------------------------------
# Define all 12 conditions
# ---------------------------------------------------------------------------
conditions = []

# 1. STDP OFF (all plasticity terms zeroed)
# NOTE: A2_plus must be tiny-nonzero (not exactly 0) because the triplet_boost
# formula divides by A2_plus.  With A2_plus=1e-20 the LTP term is effectively
# zero (1e-20 * anything ≈ 0) but avoids NaN from 0/0.
conditions.append(InvestigationConfig(
    condition_name="M1_stdp_off",
    param_overrides={
        "A2_plus": 1e-20, "A2_minus": 0,
        "A3_plus": 0, "A3_minus": 0,
        "A_het": 0, "A_split": 0,
        "split_constraint_rate": 0,
    },
    out_dir=OUT,
))

# 2. Pair-only (triplet terms zeroed)
conditions.append(InvestigationConfig(
    condition_name="M1_pair_only",
    param_overrides={"A3_plus": 0.0, "A3_minus": 0.0},
    out_dir=OUT,
))

# 3. LTP/LTD ratio sweep (fix A2_plus at default 0.008, vary A2_minus)
for a2m in [0.004, 0.006, 0.008, 0.010, 0.014, 0.020]:
    conditions.append(InvestigationConfig(
        condition_name=f"M1_A2m_{a2m:.3f}",
        param_overrides={"A2_minus": a2m},
        out_dir=OUT,
    ))

# 4. Triplet boost sweep (A3_plus = 0.0 already covered by pair-only)
for a3p in [0.001, 0.002, 0.004]:
    conditions.append(InvestigationConfig(
        condition_name=f"M1_A3p_{a3p:.3f}",
        param_overrides={"A3_plus": a3p},
        out_dir=OUT,
    ))


# ---------------------------------------------------------------------------
# Run all conditions
# ---------------------------------------------------------------------------
all_results = []
for cfg in conditions:
    print(f"\n{'='*60}")
    print(f"Running: {cfg.condition_name}")
    print(f"{'='*60}")
    result = run_investigation(cfg, extra_checkpoint_fn=m1_checkpoint_fn)
    save_condition_result(result, OUT)
    all_results.append(result)


# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------

# OSI timeseries: compare key conditions
key_results = [r for r in all_results if r["condition"] in
               ["M1_stdp_off", "M1_pair_only", "M1_A2m_0.010", "M1_A3p_0.002"]]
if key_results:
    plot_osi_timeseries(key_results, os.path.join(OUT, "osi_timeseries_key.png"),
                        title="M1: STDP variants — OSI over training")

# Dose-response: LTP/LTD ratio
ratio_results = [r for r in all_results if r["condition"].startswith("M1_A2m_")]
if ratio_results:
    x_vals = [float(r["condition"].split("_")[-1]) for r in ratio_results]
    plot_dose_response(ratio_results, x_vals, "A2_minus",
                       os.path.join(OUT, "dose_response_A2minus.png"),
                       title="M1: LTP/LTD ratio dose-response")

# Dose-response: Triplet boost
trip_results = [r for r in all_results if r["condition"].startswith("M1_A3p_")]
pair_only = [r for r in all_results if r["condition"] == "M1_pair_only"]
if trip_results and pair_only:
    trip_all = pair_only + trip_results
    x_vals = [0.0] + [float(r["condition"].split("_")[-1]) for r in trip_results]
    plot_dose_response(trip_all, x_vals, "A3_plus",
                       os.path.join(OUT, "dose_response_A3plus.png"),
                       title="M1: Triplet boost dose-response")

# Per-condition plots: pref trajectories + weight evolution
for result in all_results:
    cond_dir = os.path.join(OUT, result["condition"])
    os.makedirs(cond_dir, exist_ok=True)
    plot_pref_trajectories(result, os.path.join(cond_dir, "pref_trajectories.png"),
                           title=f"{result['condition']}: Pref orientation trajectories")
    plot_weight_evolution(result, os.path.join(cond_dir, "weight_evolution.png"),
                          title=f"{result['condition']}: Weight statistics")


# ---------------------------------------------------------------------------
# Write per-condition reports
# ---------------------------------------------------------------------------
for result in all_results:
    cond_dir = os.path.join(OUT, result["condition"])
    os.makedirs(cond_dir, exist_ok=True)
    write_condition_report(result, os.path.join(cond_dir, "report.md"))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("M1 SUMMARY")
print("=" * 60)
for r in all_results:
    s = r["summary"]
    print(f"  {r['condition']:25s}  OSI = {s['final_mean_osi']:.3f} +/- {s['final_sem_osi']:.3f}")

# Write summary JSON
summary = {r["condition"]: r["summary"] for r in all_results}
with open(os.path.join(OUT, "M1_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {OUT}/")

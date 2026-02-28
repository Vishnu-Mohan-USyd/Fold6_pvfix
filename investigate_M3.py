#!/usr/bin/env python3
"""M3: ON/OFF split mechanism investigation.

Investigates the role of the ON/OFF pathway split mechanism in OSI emergence.
Conditions:
  1. Full ablation: A_split=0, split_constraint_rate=0
  2. STDP-split only: A_split=0.2, split_constraint_rate=0
  3. Constraint only: A_split=0, split_constraint_rate=0.2
  4. Dose-response: A_split in {0.0, 0.05, 0.1, 0.2, 0.4, 0.8}
     (split_constraint_rate at default 0.2)
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "investigation_results/M3_onoff_split"
os.makedirs(OUT, exist_ok=True)


def m3_checkpoint_fn(net, segment):
    """Mechanism-specific diagnostics for ON/OFF split."""
    W = net.W.copy()
    N = net.N
    n_pix = N * N
    W_on = W[:, :n_pix]
    W_off = W[:, n_pix:]
    # Spatial overlap: pixel-wise product sum per ensemble
    overlap = (W_on * W_off).sum(axis=1)
    return {"onoff_spatial_overlap": overlap.tolist()}


# ============================================================
# Define conditions
# ============================================================
conditions = []

# 1. Full ablation: both split mechanisms OFF
conditions.append(InvestigationConfig(
    condition_name="M3_split_OFF",
    param_overrides={"A_split": 0.0, "split_constraint_rate": 0.0},
    out_dir=OUT))

# 2. STDP-split only (no constraint)
conditions.append(InvestigationConfig(
    condition_name="M3_stdp_split_only",
    param_overrides={"A_split": 0.2, "split_constraint_rate": 0.0},
    out_dir=OUT))

# 3. Constraint only (no STDP split)
conditions.append(InvestigationConfig(
    condition_name="M3_constraint_only",
    param_overrides={"A_split": 0.0, "split_constraint_rate": 0.2},
    out_dir=OUT))

# 4. Dose-response of A_split (split_constraint_rate at default 0.2)
for asplit in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]:
    conditions.append(InvestigationConfig(
        condition_name=f"M3_Asplit_{asplit:.2f}",
        param_overrides={"A_split": asplit},
        out_dir=OUT))

# ============================================================
# Run all conditions
# ============================================================
all_results = []
for cfg in conditions:
    print(f"\n{'='*60}")
    print(f"Running: {cfg.condition_name}")
    print(f"{'='*60}")
    result = run_investigation(cfg, extra_checkpoint_fn=m3_checkpoint_fn)
    save_condition_result(result, OUT)
    all_results.append(result)

# ============================================================
# Plots
# ============================================================

# Key conditions OSI timeseries
key_names = ["M3_split_OFF", "M3_stdp_split_only", "M3_constraint_only", "M3_Asplit_0.20"]
key_results = [r for r in all_results if r["condition"] in key_names]
if key_results:
    plot_osi_timeseries(key_results, os.path.join(OUT, "osi_timeseries_key.png"),
                        title="M3: ON/OFF split variants — OSI over training")

# Dose-response
dose_results = [r for r in all_results if r["condition"].startswith("M3_Asplit_")]
if dose_results:
    x_vals = [float(r["condition"].split("_")[-1]) for r in dose_results]
    plot_dose_response(dose_results, x_vals, "A_split",
                       os.path.join(OUT, "dose_response_Asplit.png"),
                       title="M3: ON/OFF split dose-response")

# ON/OFF correlation vs OSI scatter (all ensembles, all conditions)
fig, ax = plt.subplots(figsize=(7, 5))
for r in all_results:
    for si, sr in enumerate(r["seeds"]):
        final = sr["checkpoints"][-1]
        osi_arr = np.array(final["osi"])
        corr_arr = np.array(final["onoff_corr"])
        ax.scatter(corr_arr, osi_arr, alpha=0.3, s=15,
                   label=r["condition"] if si == 0 else "")
ax.set_xlabel("ON/OFF weight correlation")
ax.set_ylabel("OSI")
ax.set_title("M3: ON/OFF correlation vs OSI (all ensembles, all conditions)")
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "onoff_corr_vs_osi_scatter.png"), dpi=150)
plt.close(fig)

# Spatial overlap vs OSI scatter
fig, ax = plt.subplots(figsize=(7, 5))
for r in all_results:
    for si, sr in enumerate(r["seeds"]):
        final = sr["checkpoints"][-1]
        osi_arr = np.array(final["osi"])
        overlap_arr = np.array(final["onoff_spatial_overlap"])
        ax.scatter(overlap_arr, osi_arr, alpha=0.3, s=15,
                   label=r["condition"] if si == 0 else "")
ax.set_xlabel("ON/OFF spatial overlap (sum W_on * W_off)")
ax.set_ylabel("OSI")
ax.set_title("M3: ON/OFF spatial overlap vs OSI (all ensembles, all conditions)")
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "onoff_overlap_vs_osi_scatter.png"), dpi=150)
plt.close(fig)

# Per-condition reports
for result in all_results:
    cond_dir = os.path.join(OUT, result["condition"])
    os.makedirs(cond_dir, exist_ok=True)
    write_condition_report(result, os.path.join(cond_dir, "report.md"))

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("M3 SUMMARY")
print("=" * 60)
for r in all_results:
    s = r["summary"]
    print(f"  {r['condition']:30s}  OSI = {s['final_mean_osi']:.3f} +/- {s['final_sem_osi']:.3f}")

summary = {r["condition"]: r["summary"] for r in all_results}
with open(os.path.join(OUT, "M3_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {OUT}/")

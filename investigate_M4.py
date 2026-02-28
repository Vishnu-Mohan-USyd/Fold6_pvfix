#!/usr/bin/env python3
"""M4: PV inhibition + iSTDP mechanism investigation.

Investigates the role of PV interneuron inhibition and inhibitory STDP
(iSTDP) in OSI emergence.

Conditions (9 total):
  1. Full ablation — w_pv_e=0, pv_inhib_plastic=False
  2. Fixed PV (no iSTDP) — w_pv_e=1.0, pv_inhib_plastic=False
  3. Plastic PV from low — w_pv_e=0.1, pv_inhib_plastic=True
  4. Dose-response (fixed PV, no plasticity):
       w_pv_e in {0.0, 0.25, 0.5, 1.0, 2.0, 4.0}, pv_inhib_plastic=False

Level-3 deep analysis: PV->E weight stats at each checkpoint.
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

OUT = "investigation_results/M4_pv_inhibition"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Mechanism-specific checkpoint diagnostics
# ---------------------------------------------------------------------------

def m4_checkpoint_fn(net, segment):
    """PV inhibition diagnostics at each checkpoint.

    Metrics:
      - pv_e_weight_mean/std/max : PV->E weight statistics
      - v1_rate_cv              : CV of peak firing rates across ensembles
    """
    metrics = {}

    # PV->E weight stats
    W_pv_e = net.W_pv_e.copy()
    metrics["pv_e_weight_mean"] = float(W_pv_e.mean())
    metrics["pv_e_weight_std"] = float(W_pv_e.std())
    metrics["pv_e_weight_max"] = float(W_pv_e.max())

    return metrics


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

conditions = []

# 1. Full ablation
conditions.append(InvestigationConfig(
    condition_name="M4_pv_OFF",
    param_overrides={"w_pv_e": 0.0, "pv_inhib_plastic": False},
    out_dir=OUT))

# 2. Fixed PV (no plasticity)
conditions.append(InvestigationConfig(
    condition_name="M4_fixed_pv",
    param_overrides={"w_pv_e": 1.0, "pv_inhib_plastic": False},
    out_dir=OUT))

# 3. Plastic from low initial
conditions.append(InvestigationConfig(
    condition_name="M4_plastic_low",
    param_overrides={"w_pv_e": 0.1, "pv_inhib_plastic": True},
    out_dir=OUT))

# 4. Dose-response (fixed, no plasticity)
for wpv in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
    conditions.append(InvestigationConfig(
        condition_name=f"M4_wpv_{wpv:.2f}",
        param_overrides={"w_pv_e": wpv, "pv_inhib_plastic": False},
        out_dir=OUT))


# ---------------------------------------------------------------------------
# Run all conditions
# ---------------------------------------------------------------------------

all_results = []
for cfg in conditions:
    print(f"\n{'='*60}")
    print(f"Running: {cfg.condition_name}")
    print(f"{'='*60}")
    result = run_investigation(cfg, extra_checkpoint_fn=m4_checkpoint_fn)
    save_condition_result(result, OUT)
    all_results.append(result)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# Key conditions OSI timeseries
key_names = ["M4_pv_OFF", "M4_fixed_pv", "M4_plastic_low"]
key_results = [r for r in all_results if r["condition"] in key_names]
if key_results:
    plot_osi_timeseries(key_results, os.path.join(OUT, "osi_timeseries_key.png"),
                        title="M4: PV inhibition variants — OSI over training")

# Dose-response
dose_results = [r for r in all_results if r["condition"].startswith("M4_wpv_")]
if dose_results:
    x_vals = [float(r["condition"].split("_")[-1]) for r in dose_results]
    plot_dose_response(dose_results, x_vals, "w_pv_e",
                       os.path.join(OUT, "dose_response_wpv.png"),
                       title="M4: PV→E weight dose-response")

# Firing rate comparison: PV OFF vs PV ON (w_pv_e=1.0)
fig, ax = plt.subplots(figsize=(8, 5))
for r in all_results:
    if r["condition"] in ["M4_pv_OFF", "M4_wpv_1.00"]:
        sr = r["seeds"][0]  # first seed
        ckpts = sr["checkpoints"]
        segs = [c["segment"] for c in ckpts]
        rates = [c["mean_rate_hz"] for c in ckpts]
        ax.plot(segs, rates, label=r["condition"])
ax.set_xlabel("Training segment")
ax.set_ylabel("Mean V1 firing rate (Hz)")
ax.set_title("M4: V1 firing rate — PV ON vs OFF")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "firing_rate_comparison.png"), dpi=150)
plt.close(fig)

# PV->E weight evolution for plastic condition
plastic_results = [r for r in all_results if r["condition"] == "M4_plastic_low"]
if plastic_results:
    pr = plastic_results[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for si, sr in enumerate(pr["seeds"]):
        ckpts = sr["checkpoints"]
        segs = [c["segment"] for c in ckpts]
        pv_means = [c.get("pv_e_weight_mean", 0) for c in ckpts]
        ax.plot(segs, pv_means, label=f"seed={sr['seed']}", alpha=0.8)
    ax.set_xlabel("Training segment")
    ax.set_ylabel("Mean PV→E weight")
    ax.set_title("M4: PV→E weight evolution (plastic from low)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pv_weight_evolution.png"), dpi=150)
    plt.close(fig)

# Pref trajectory and weight evolution for key conditions
for cname in key_names:
    cresults = [r for r in all_results if r["condition"] == cname]
    if cresults:
        plot_pref_trajectories(cresults[0],
                               os.path.join(OUT, f"pref_traj_{cname}.png"),
                               title=f"{cname}: Preferred orientation trajectories")
        plot_weight_evolution(cresults[0],
                              os.path.join(OUT, f"weight_evol_{cname}.png"),
                              title=f"{cname}: Weight statistics")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

for result in all_results:
    cond_dir = os.path.join(OUT, result["condition"])
    os.makedirs(cond_dir, exist_ok=True)
    write_condition_report(result, os.path.join(cond_dir, "report.md"))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("M4 SUMMARY")
print("=" * 60)
for r in all_results:
    s = r["summary"]
    print(f"  {r['condition']:25s}  OSI = {s['final_mean_osi']:.3f} +/- {s['final_sem_osi']:.3f}")

summary = {r["condition"]: r["summary"] for r in all_results}
with open(os.path.join(OUT, "M4_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {OUT}/")

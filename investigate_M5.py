#!/usr/bin/env python3
"""M5: SOM lateral inhibition mechanism investigation."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osi_investigation_harness import *
import numpy as np

OUT = "investigation_results/M5_som_inhibition"
os.makedirs(OUT, exist_ok=True)

def m5_checkpoint_fn(net, segment):
    """Mechanism-specific diagnostics for SOM."""
    # Pairwise pref orientation differences (measure diversity directly)
    return {}  # Standard metrics already capture R and max_gap

# Conditions
conditions = []

# Ablation
conditions.append(InvestigationConfig(
    condition_name="M5_som_OFF",
    param_overrides={"w_som_e": 0.0},
    out_dir=OUT))

# Self-inhibition only (very narrow spread)
conditions.append(InvestigationConfig(
    condition_name="M5_narrow_som",
    param_overrides={"som_out_sigma": 0.01},
    out_dir=OUT))

# Broad inhibition
conditions.append(InvestigationConfig(
    condition_name="M5_broad_som",
    param_overrides={"som_out_sigma": 3.0},
    out_dir=OUT))

# Dose-response
for wsom in [0.0, 0.01, 0.025, 0.05, 0.1, 0.2]:
    conditions.append(InvestigationConfig(
        condition_name=f"M5_wsom_{wsom:.3f}",
        param_overrides={"w_som_e": wsom},
        out_dir=OUT))

# Run all
all_results = []
for cfg in conditions:
    print(f"\n{'='*60}")
    print(f"Running: {cfg.condition_name}")
    print(f"{'='*60}")
    result = run_investigation(cfg, extra_checkpoint_fn=m5_checkpoint_fn)
    save_condition_result(result, OUT)
    all_results.append(result)

# --- Plots ---
# OSI timeseries
key_names = ["M5_som_OFF", "M5_narrow_som", "M5_broad_som", "M5_wsom_0.050"]
key_results = [r for r in all_results if r["condition"] in key_names]
if key_results:
    plot_osi_timeseries(key_results, os.path.join(OUT, "osi_timeseries_key.png"),
                        title="M5: SOM variants — OSI over training")

# Dose-response (OSI)
dose_results = [r for r in all_results if r["condition"].startswith("M5_wsom_")]
if dose_results:
    x_vals = [float(r["condition"].split("_")[-1]) for r in dose_results]
    plot_dose_response(dose_results, x_vals, "w_som_e",
                       os.path.join(OUT, "dose_response_wsom_osi.png"),
                       title="M5: SOM→E weight dose-response (OSI)")

# Diversity dose-response: plot resultant R and max_gap vs w_som_e
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if dose_results:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x_vals = []
    R_means = []
    gap_means = []
    for r in dose_results:
        x_vals.append(float(r["condition"].split("_")[-1]))
        Rs = [sr["checkpoints"][-1]["pref_resultant_R"] for sr in r["seeds"]]
        gaps = [sr["checkpoints"][-1]["pref_max_gap"] for sr in r["seeds"]]
        R_means.append(np.mean(Rs))
        gap_means.append(np.mean(gaps))
    ax1.plot(x_vals, R_means, "o-")
    ax1.set_xlabel("w_som_e")
    ax1.set_ylabel("Pref orientation resultant R")
    ax1.set_title("Lower R = more diverse preferences")
    ax1.grid(True, alpha=0.3)
    ax2.plot(x_vals, gap_means, "o-")
    ax2.set_xlabel("w_som_e")
    ax2.set_ylabel("Max circular gap (deg)")
    ax2.set_title("Lower gap = better coverage")
    ax2.grid(True, alpha=0.3)
    fig.suptitle("M5: SOM effect on orientation diversity")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diversity_dose_response.png"), dpi=150)
    plt.close(fig)

# Reports
for result in all_results:
    write_condition_report(result, os.path.join(OUT, result["condition"], "report.md"))

# Summary
print("\n" + "="*60)
print("M5 SUMMARY")
print("="*60)
for r in all_results:
    s = r["summary"]
    final_R = np.mean([sr["checkpoints"][-1]["pref_resultant_R"] for sr in r["seeds"]])
    print(f"  {r['condition']:25s}  OSI = {s['final_mean_osi']:.3f} +/- {s['final_sem_osi']:.3f}  R={final_R:.3f}")

summary = {r["condition"]: r["summary"] for r in all_results}
with open(os.path.join(OUT, "M5_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {OUT}/")

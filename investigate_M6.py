#!/usr/bin/env python3
"""M6: Thalamocortical STP mechanism investigation.

Investigates the role of thalamocortical short-term plasticity (STP)
in OSI emergence. Conditions:
  - Full ablation (both paths off)
  - E-path only STP
  - PV-path only STP
  - Dose-response: tc_stp_u in {0.0, 0.02, 0.05, 0.1, 0.2, 0.5}
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osi_investigation_harness import *
import numpy as np

OUT = "investigation_results/M6_tc_stp"
os.makedirs(OUT, exist_ok=True)


def m6_checkpoint_fn(net, segment):
    """Mechanism-specific diagnostics for TC STP."""
    metrics = {}
    # E-pathway STP resource
    if net.tc_stp_x is not None:
        metrics["stp_x_mean"] = float(net.tc_stp_x.mean())
        metrics["stp_x_min"] = float(net.tc_stp_x.min())
        metrics["stp_x_std"] = float(net.tc_stp_x.std())
    else:
        metrics["stp_x_mean"] = 1.0
        metrics["stp_x_min"] = 1.0
        metrics["stp_x_std"] = 0.0
    # PV-pathway STP resource
    if net.tc_stp_x_pv is not None:
        metrics["stp_x_pv_mean"] = float(net.tc_stp_x_pv.mean())
    else:
        metrics["stp_x_pv_mean"] = 1.0
    # Effective weight: W * stp_x (for E pathway)
    if net.tc_stp_x is not None:
        W_eff = net.W * net.tc_stp_x
        metrics["eff_weight_mean"] = float(W_eff.mean())
        metrics["compression_ratio"] = float(W_eff.mean() / (net.W.mean() + 1e-12))
    return metrics


# ---- Conditions ----
conditions = []

# Full ablation
conditions.append(InvestigationConfig(
    condition_name="M6_stp_OFF",
    param_overrides={"tc_stp_enabled": False, "tc_stp_pv_enabled": False},
    out_dir=OUT))

# E-path only
conditions.append(InvestigationConfig(
    condition_name="M6_stp_E_only",
    param_overrides={"tc_stp_enabled": True, "tc_stp_pv_enabled": False},
    out_dir=OUT))

# PV-path only
conditions.append(InvestigationConfig(
    condition_name="M6_stp_PV_only",
    param_overrides={"tc_stp_enabled": False, "tc_stp_pv_enabled": True},
    out_dir=OUT))

# Dose-response (both paths enabled, varying depletion strength)
for u in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]:
    conditions.append(InvestigationConfig(
        condition_name=f"M6_stpu_{u:.2f}",
        param_overrides={"tc_stp_enabled": True, "tc_stp_pv_enabled": True,
                         "tc_stp_u": u, "tc_stp_pv_u": u},
        out_dir=OUT))

# ---- Run all conditions ----
all_results = []
for cfg in conditions:
    print(f"\n{'='*60}")
    print(f"Running: {cfg.condition_name}")
    print(f"{'='*60}")
    result = run_investigation(cfg, extra_checkpoint_fn=m6_checkpoint_fn)
    save_condition_result(result, OUT)
    all_results.append(result)

# ---- Plots ----

# 1) Key conditions OSI timeseries
key_names = ["M6_stp_OFF", "M6_stp_E_only", "M6_stp_PV_only", "M6_stpu_0.05"]
key_results = [r for r in all_results if r["condition"] in key_names]
if key_results:
    plot_osi_timeseries(key_results, os.path.join(OUT, "osi_timeseries_key.png"),
                        title="M6: TC STP variants — OSI over training")

# 2) Dose-response curve
dose_results = [r for r in all_results if r["condition"].startswith("M6_stpu_")]
if dose_results:
    x_vals = [float(r["condition"].split("_")[-1]) for r in dose_results]
    plot_dose_response(dose_results, x_vals, "tc_stp_u",
                       os.path.join(OUT, "dose_response_stpu.png"),
                       title="M6: STP depletion (u) dose-response")

# 3) STP diagnostics: saturation + resource levels
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Weight saturation: STP ON vs OFF
for cond_name in ["M6_stp_OFF", "M6_stpu_0.05"]:
    res = [r for r in all_results if r["condition"] == cond_name]
    if res:
        r = res[0]
        sr = r["seeds"][0]
        ckpts = sr["checkpoints"]
        segs = [c["segment"] for c in ckpts]
        sat = [np.mean(c["sat_frac"]) for c in ckpts]
        ax1.plot(segs, sat, label=cond_name)
ax1.set_xlabel("Training segment")
ax1.set_ylabel("Mean weight saturation fraction")
ax1.set_title("Weight saturation: STP ON vs OFF")
ax1.legend()
ax1.grid(True, alpha=0.3)

# STP resource level over time
for cond_name in ["M6_stpu_0.02", "M6_stpu_0.05", "M6_stpu_0.20"]:
    res = [r for r in all_results if r["condition"] == cond_name]
    if res:
        r = res[0]
        sr = r["seeds"][0]
        ckpts = sr["checkpoints"]
        segs = [c["segment"] for c in ckpts]
        stp_x = [c.get("stp_x_mean", 1.0) for c in ckpts]
        ax2.plot(segs, stp_x, label=cond_name)
ax2.set_xlabel("Training segment")
ax2.set_ylabel("Mean STP resource (x)")
ax2.set_title("STP resource level over training")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle("M6: TC STP diagnostics")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "stp_diagnostics.png"), dpi=150)
plt.close(fig)

# ---- Per-condition reports ----
for result in all_results:
    cond_dir = os.path.join(OUT, result["condition"])
    os.makedirs(cond_dir, exist_ok=True)
    write_condition_report(result, os.path.join(cond_dir, "report.md"))

# ---- Summary ----
print("\n" + "="*60)
print("M6 SUMMARY")
print("="*60)
for r in all_results:
    s = r["summary"]
    print(f"  {r['condition']:25s}  OSI = {s['final_mean_osi']:.3f} +/- {s['final_sem_osi']:.3f}")

summary = {r["condition"]: r["summary"] for r in all_results}
with open(os.path.join(OUT, "M6_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {OUT}/")

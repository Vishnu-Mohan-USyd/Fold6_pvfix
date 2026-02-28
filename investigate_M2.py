#!/usr/bin/env python3
"""M2: Heterosynaptic depression mechanism investigation.

Sweeps A_het across 8 values (0.0 to 0.128) to determine whether
heterosynaptic depression is necessary, facilitating, or dispensable
for OSI emergence.  Includes Level 3 deep diagnostics:
  - Weight saturation fraction per ensemble
  - Weight entropy per ensemble
  - Winner vs loser synapse tracking (top/bottom 25%)
  - Total weight per ensemble
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osi_investigation_harness import *
import numpy as np

OUT = "investigation_results/M2_heterosynaptic"
os.makedirs(OUT, exist_ok=True)


# ======================================================================
# Level 3 extra checkpoint function
# ======================================================================

def m2_checkpoint_fn(net, segment):
    """Mechanism-specific diagnostics for heterosynaptic depression.

    At each checkpoint, compute per-ensemble:
      - sat_frac_ens: fraction of synapses > 0.9 * w_max
      - w_entropy_ens: Shannon entropy of normalised weight histogram
      - winner_mean_w: mean weight of top 25% synapses (by current weight)
      - loser_mean_w: mean weight of bottom 25% synapses
      - w_total_ens: total synaptic weight
    """
    W = net.W.copy()
    M = net.M
    w_max = net.p.w_max
    n_syn = W.shape[1]
    q25 = max(1, int(n_syn * 0.25))

    winner_mean = []
    loser_mean = []
    sat_frac_ens = []
    w_entropy_ens = []
    w_total_ens = []

    for m in range(M):
        row = W[m]
        w_sorted = np.sort(row)

        # Winner / loser analysis
        loser_mean.append(float(w_sorted[:q25].mean()))
        winner_mean.append(float(w_sorted[-q25:].mean()))

        # Saturation fraction
        sat_frac_ens.append(float((row > 0.9 * w_max).sum() / n_syn))

        # Shannon entropy of weight histogram
        n_bins = 30
        hist, _ = np.histogram(row, bins=n_bins, range=(0, w_max + 1e-9))
        p_hist = hist.astype(np.float64) / hist.sum()
        p_hist = p_hist[p_hist > 0]
        entropy = float(-np.sum(p_hist * np.log2(p_hist))) if len(p_hist) > 0 else 0.0
        w_entropy_ens.append(entropy)

        # Total weight
        w_total_ens.append(float(row.sum()))

    return {
        "winner_mean_w": winner_mean,
        "loser_mean_w": loser_mean,
        "sat_frac_ens": sat_frac_ens,
        "w_entropy_ens": w_entropy_ens,
        "w_total_ens": w_total_ens,
    }


# ======================================================================
# Define conditions
# ======================================================================

a_het_values = [0.0, 0.008, 0.016, 0.032, 0.048, 0.064, 0.096, 0.128]
conditions = []
for ah in a_het_values:
    name = f"M2_Ahet_{ah:.3f}" if ah > 0 else "M2_Ahet_OFF"
    conditions.append(InvestigationConfig(
        condition_name=name,
        param_overrides={"A_het": ah},
        out_dir=OUT,
    ))


# ======================================================================
# Run all conditions
# ======================================================================

t_start = time.time()
all_results = []
for ci, cfg in enumerate(conditions):
    print(f"\n{'='*60}")
    print(f"Running condition {ci+1}/{len(conditions)}: {cfg.condition_name}")
    print(f"{'='*60}")
    result = run_investigation(cfg, extra_checkpoint_fn=m2_checkpoint_fn)
    save_condition_result(result, OUT)
    all_results.append(result)
    elapsed = time.time() - t_start
    print(f"  [Elapsed: {elapsed/60:.1f} min]")


# ======================================================================
# Generate plots
# ======================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. OSI timeseries (all conditions)
plot_osi_timeseries(all_results, os.path.join(OUT, "osi_timeseries_all.png"),
                    title="M2: Heterosynaptic depression — OSI over training")

# 2. Dose-response curve
plot_dose_response(all_results, a_het_values, "A_het",
                   os.path.join(OUT, "dose_response_Ahet.png"),
                   title="M2: Heterosynaptic depression dose-response")

# 3. Winner/loser trajectory plots for ablation vs baseline vs strong
key_conditions = [
    ("M2_Ahet_OFF", "A_het=0 (ablation)"),
    ("M2_Ahet_0.032", "A_het=0.032 (baseline)"),
    ("M2_Ahet_0.128", "A_het=0.128 (strong)"),
]

for cond_name, label in key_conditions:
    res_list = [r for r in all_results if r["condition"] == cond_name]
    if not res_list:
        continue
    r = res_list[0]
    seed_res = r["seeds"][0]
    ckpts = seed_res["checkpoints"]
    segs = [c["segment"] for c in ckpts]
    w_win = [np.mean(c.get("winner_mean_w", [0])) for c in ckpts]
    w_lose = [np.mean(c.get("loser_mean_w", [0])) for c in ckpts]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(segs, w_win, "r-o", markersize=3, label="Winners (top 25%)")
    ax.plot(segs, w_lose, "b-o", markersize=3, label="Losers (bottom 25%)")
    ax.set_xlabel("Training segment")
    ax.set_ylabel("Mean weight")
    ax.set_title(f"Winner/Loser trajectories ({label})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"winner_loser_{cond_name}.png"), dpi=150)
    plt.close(fig)

# 4. Comparative winner/loser overlay (ablation vs baseline)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, (cond_name, label) in zip(axes, [("M2_Ahet_OFF", "A_het=0"), ("M2_Ahet_0.032", "A_het=0.032")]):
    res_list = [r for r in all_results if r["condition"] == cond_name]
    if not res_list:
        continue
    r = res_list[0]
    # Average across all seeds
    for si, seed_res in enumerate(r["seeds"]):
        ckpts = seed_res["checkpoints"]
        segs = [c["segment"] for c in ckpts]
        w_win = [np.mean(c.get("winner_mean_w", [0])) for c in ckpts]
        w_lose = [np.mean(c.get("loser_mean_w", [0])) for c in ckpts]
        alpha = 1.0 if si == 0 else 0.4
        lw = 2 if si == 0 else 1
        ax.plot(segs, w_win, "r-", alpha=alpha, linewidth=lw,
                label="Winners" if si == 0 else None)
        ax.plot(segs, w_lose, "b-", alpha=alpha, linewidth=lw,
                label="Losers" if si == 0 else None)
    ax.set_xlabel("Training segment")
    ax.set_ylabel("Mean weight")
    ax.set_title(label)
    ax.legend()
    ax.grid(True, alpha=0.3)
fig.suptitle("M2: Winner/Loser weight separation", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "winner_loser_comparison.png"), dpi=150)
plt.close(fig)

# 5. Saturation fraction over training for key conditions
fig, ax = plt.subplots(figsize=(8, 5))
for cond_name, label in key_conditions:
    res_list = [r for r in all_results if r["condition"] == cond_name]
    if not res_list:
        continue
    r = res_list[0]
    seed_res = r["seeds"][0]
    ckpts = seed_res["checkpoints"]
    segs = [c["segment"] for c in ckpts]
    sat = [np.mean(c.get("sat_frac_ens", c.get("sat_frac", [0]))) for c in ckpts]
    ax.plot(segs, sat, "-o", markersize=3, label=label)
ax.set_xlabel("Training segment")
ax.set_ylabel("Mean saturation fraction (>0.9*w_max)")
ax.set_title("M2: Weight saturation over training")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "saturation_comparison.png"), dpi=150)
plt.close(fig)

# 6. Weight entropy over training for key conditions
fig, ax = plt.subplots(figsize=(8, 5))
for cond_name, label in key_conditions:
    res_list = [r for r in all_results if r["condition"] == cond_name]
    if not res_list:
        continue
    r = res_list[0]
    seed_res = r["seeds"][0]
    ckpts = seed_res["checkpoints"]
    segs = [c["segment"] for c in ckpts]
    ent = [np.mean(c.get("w_entropy_ens", c.get("w_entropy", [0]))) for c in ckpts]
    ax.plot(segs, ent, "-o", markersize=3, label=label)
ax.set_xlabel("Training segment")
ax.set_ylabel("Mean weight entropy (bits)")
ax.set_title("M2: Weight entropy over training")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "entropy_comparison.png"), dpi=150)
plt.close(fig)

# 7. Weight evolution for ablation vs baseline
for r in all_results:
    if r["condition"] in ["M2_Ahet_OFF", "M2_Ahet_0.032"]:
        plot_weight_evolution(r, os.path.join(OUT, f"weight_evo_{r['condition']}.png"),
                              title=f"Weight evolution: {r['condition']}")

# 8. Preference trajectories for ablation vs baseline
for r in all_results:
    if r["condition"] in ["M2_Ahet_OFF", "M2_Ahet_0.032"]:
        plot_pref_trajectories(r, os.path.join(OUT, f"pref_traj_{r['condition']}.png"),
                               title=f"Pref orientation: {r['condition']}")

# 9. Write per-condition reports
for result in all_results:
    cond_dir = os.path.join(OUT, result["condition"])
    os.makedirs(cond_dir, exist_ok=True)
    write_condition_report(result, os.path.join(cond_dir, "report.md"))


# ======================================================================
# Summary
# ======================================================================

print("\n" + "=" * 60)
print("M2 INVESTIGATION SUMMARY")
print("=" * 60)
print(f"{'Condition':25s}  {'A_het':>6s}  {'OSI mean':>9s}  {'OSI SEM':>8s}")
print("-" * 55)
for r, ah in zip(all_results, a_het_values):
    s = r["summary"]
    print(f"  {r['condition']:23s}  {ah:6.3f}  {s['final_mean_osi']:9.4f}  {s['final_sem_osi']:8.4f}")

# Classify role of heterosynaptic depression
baseline_osi = None
ablation_osi = None
for r, ah in zip(all_results, a_het_values):
    s = r["summary"]
    if ah == 0.0:
        ablation_osi = s["final_mean_osi"]
    if ah == 0.032:
        baseline_osi = s["final_mean_osi"]

if ablation_osi is not None and baseline_osi is not None:
    osi_drop = baseline_osi - ablation_osi
    pct_drop = 100.0 * osi_drop / baseline_osi if baseline_osi > 0 else float("nan")
    print(f"\nAblation effect: baseline OSI = {baseline_osi:.4f}, ablation OSI = {ablation_osi:.4f}")
    print(f"  Drop = {osi_drop:.4f} ({pct_drop:.1f}% of baseline)")
    if ablation_osi > 0.5:
        if pct_drop < 10:
            verdict = "DISPENSABLE — OSI remains high without heterosynaptic depression"
        else:
            verdict = "FACILITATING — OSI degrades moderately but remains above 0.5"
    elif ablation_osi > 0.3:
        verdict = "IMPORTANT — OSI drops substantially but some selectivity remains"
    else:
        verdict = "NECESSARY — OSI collapses without heterosynaptic depression"
    print(f"  Verdict: {verdict}")

# Save summary
summary = {}
for r, ah in zip(all_results, a_het_values):
    summary[r["condition"]] = {
        **r["summary"],
        "A_het": ah,
    }
with open(os.path.join(OUT, "M2_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

total_time = time.time() - t_start
print(f"\nTotal wall time: {total_time/60:.1f} min")
print(f"Results saved to {OUT}/")

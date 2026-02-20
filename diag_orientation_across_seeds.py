#!/usr/bin/env python3
"""Measure preferred orientation distribution after Phase A across multiple seeds.

For each seed:
  1. Create M=32 network with seeded init
  2. Run 300 segments of STDP learning (Phase A)
  3. Evaluate tuning curves
  4. Bin preferred orientations of tuned ensembles

Then aggregate bin counts across all seeds.
"""
import argparse
import os
import sys
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biologically_plausible_v1_stdp import (
    Params, RgcLgnV1Network, compute_osi,
)


def run_phase_a(seed: int, M: int = 32, n_segments: int = 300):
    """Run Phase A for one seed. Returns (pref_deg, osi, rates_hz)."""
    p = Params(N=8, M=M, seed=seed, v1_bias_eta=0.0, train_contrast=2.0)
    net = RgcLgnV1Network(p, init_mode="random")

    # Low-discrepancy training schedule (golden ratio)
    phi = (1.0 + math.sqrt(5)) / 2.0
    theta_step = 180.0 / phi
    rng = np.random.default_rng(seed)
    theta0 = float(rng.uniform(0, 180))

    for s in range(n_segments):
        th = (theta0 + s * theta_step) % 180.0
        net.run_segment(th, plastic=True, contrast=2.0)
        if (s + 1) % 100 == 0:
            print(f"    [seed {seed}] seg {s+1}/{n_segments}")

    # Evaluate
    thetas = np.linspace(0, 180 - 180/18, 18)  # 10° bins
    rates = net.evaluate_tuning(thetas, repeats=5, contrast=2.0)
    osi, pref = compute_osi(rates, thetas)

    return pref, osi, rates, thetas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_pref_hist"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--segments", type=int, default=300)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bin_deg = 15
    bins = np.arange(0.0, 180.0 + bin_deg, bin_deg)
    bin_centers = bins[:-1] + bin_deg / 2.0

    all_prefs = []
    all_osis = []
    per_seed_counts = []
    osi_thresh = 0.3

    for seed in range(1, args.seeds + 1):
        print(f"  === Seed {seed}/{args.seeds} ===")
        pref, osi, rates, thetas = run_phase_a(seed, M=args.M, n_segments=args.segments)

        tuned = osi >= osi_thresh
        n_tuned = int(tuned.sum())
        prefs_tuned = pref[tuned]
        mean_osi = float(osi.mean())
        mean_osi_tuned = float(osi[tuned].mean()) if n_tuned > 0 else 0.0

        all_prefs.extend(prefs_tuned.tolist())
        all_osis.extend(osi[tuned].tolist())

        counts, _ = np.histogram(prefs_tuned, bins=bins)
        per_seed_counts.append(counts)

        print(f"    tuned={n_tuned}/{args.M}, mean_OSI={mean_osi:.3f}, "
              f"mean_OSI(tuned)={mean_osi_tuned:.3f}")
        print(f"    bin counts: {counts.tolist()}")

    all_prefs = np.array(all_prefs)
    all_osis = np.array(all_osis)
    per_seed_counts = np.array(per_seed_counts)  # (n_seeds, n_bins)
    total_counts = per_seed_counts.sum(axis=0)
    mean_counts = per_seed_counts.mean(axis=0)
    std_counts = per_seed_counts.std(axis=0)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  AGGREGATE: {args.seeds} seeds x M={args.M}, {args.segments} segments")
    print(f"  OSI threshold = {osi_thresh}")
    print(f"  Total tuned ensembles: {len(all_prefs)} / {args.seeds * args.M}")
    print(f"{'='*70}")
    print(f"\n  {'Bin':>10s}  {'Total':>6s}  {'Mean±Std':>12s}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*12}")
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        print(f"  {lo:5.0f}-{hi:4.0f}°  {total_counts[i]:6d}  "
              f"{mean_counts[i]:5.1f} ± {std_counts[i]:4.1f}")
    print(f"  {'TOTAL':>10s}  {total_counts.sum():6d}")

    # Uniformity statistics
    expected = total_counts.sum() / len(total_counts)
    chi2 = float(np.sum((total_counts - expected)**2 / expected))
    dof = len(total_counts) - 1
    print(f"\n  Expected per bin (uniform): {expected:.1f}")
    print(f"  Chi-squared = {chi2:.2f} (dof={dof})")
    # Rough p-value via normal approximation for chi2
    z = (chi2 - dof) / np.sqrt(2 * dof)
    print(f"  z-score = {z:.2f} (>2 = significant non-uniformity)")

    # Circular stats
    th_rad = np.deg2rad(all_prefs * 2)  # doubled angle
    R = np.abs(np.mean(np.exp(1j * th_rad)))
    mu = np.degrees(np.angle(np.mean(np.exp(1j * th_rad)))) / 2 % 180
    print(f"\n  Circular resultant R = {R:.4f} (0=uniform, 1=all same)")
    print(f"  Circular mean direction = {mu:.1f}°")

    # --- Plot 1: Aggregate histogram ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(bin_centers, total_counts, width=bin_deg * 0.85, color="steelblue",
           edgecolor="k", alpha=0.85)
    ax.axhline(expected, color="red", ls="--", lw=1.5,
               label=f"Uniform expected = {expected:.1f}")
    ax.set_xlabel("Preferred orientation (°)", fontsize=12)
    ax.set_ylabel(f"Count (OSI ≥ {osi_thresh})", fontsize=12)
    ax.set_title(f"Preferred orientation after Phase A ({args.segments} segs)\n"
                 f"{args.seeds} seeds × M={args.M} | "
                 f"R={R:.3f}, χ²={chi2:.1f} (dof={dof})",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 15))
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "aggregate_pref_hist.png"), dpi=200)
    plt.close(fig)
    print(f"\n  Saved aggregate_pref_hist.png")

    # --- Plot 2: Per-seed heatmap ---
    fig, ax = plt.subplots(figsize=(10, max(3, args.seeds * 0.4 + 1)))
    im = ax.imshow(per_seed_counts, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_xticks(np.arange(len(bin_centers)))
    ax.set_xticklabels([f"{c:.0f}°" for c in bin_centers], fontsize=8, rotation=45)
    ax.set_yticks(np.arange(args.seeds))
    ax.set_yticklabels([f"Seed {s+1}" for s in range(args.seeds)], fontsize=9)
    ax.set_xlabel("Preferred orientation bin")
    ax.set_ylabel("Seed")
    ax.set_title(f"Per-seed bin counts (M={args.M}, {args.segments} segs)",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Count", fraction=0.03)
    # Annotate cells
    for i in range(args.seeds):
        for j in range(len(bin_centers)):
            v = per_seed_counts[i, j]
            color = "white" if v > per_seed_counts.max() * 0.6 else "black"
            ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "per_seed_heatmap.png"), dpi=200)
    plt.close(fig)
    print(f"  Saved per_seed_heatmap.png")

    # --- Plot 3: Mean ± std bar chart ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(bin_centers, mean_counts, width=bin_deg * 0.85, color="steelblue",
           edgecolor="k", alpha=0.85, yerr=std_counts, capsize=3, ecolor="black")
    ax.axhline(expected / args.seeds, color="red", ls="--", lw=1.5,
               label=f"Uniform expected = {expected/args.seeds:.1f}/seed")
    ax.set_xlabel("Preferred orientation (°)", fontsize=12)
    ax.set_ylabel(f"Mean count per seed ± std", fontsize=12)
    ax.set_title(f"Mean ± std bin counts across {args.seeds} seeds\n"
                 f"M={args.M}, {args.segments} segments, OSI ≥ {osi_thresh}",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 15))
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "mean_std_hist.png"), dpi=200)
    plt.close(fig)
    print(f"  Saved mean_std_hist.png")

    # Save raw data
    np.savez(os.path.join(args.out, "seed_data.npz"),
             per_seed_counts=per_seed_counts,
             bins=bins, bin_centers=bin_centers,
             all_prefs=all_prefs, all_osis=all_osis,
             total_counts=total_counts)
    print(f"  Saved seed_data.npz")


if __name__ == "__main__":
    main()

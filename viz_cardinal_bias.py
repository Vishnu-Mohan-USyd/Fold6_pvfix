#!/usr/bin/env python3
"""Visualize cardinal bias diagnostic results.

Reads the saved JSON from diag_cardinal_bias.py and creates publication-quality
figures showing V1 spike counts per orientation for random vs seeded init.
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = "/tmp/diag_cardinal/viz"
DATA_PATH = "/tmp/diag_cardinal/cardinal_bias_results.json"

THETAS = np.arange(0.0, 180.0, 10.0)

# Cardinal: within 10° of 0° or 90°; Oblique: within 10° of 45° or 135°
def is_cardinal(th):
    return min(th % 180, abs(th % 180 - 90), abs(th % 180 - 180)) <= 10
def is_oblique(th):
    return min(abs(th % 180 - 45), abs(th % 180 - 135)) <= 10


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def fig1_spikes_per_theta(data):
    """Per-orientation V1 spike counts for each seed, random vs seeded."""
    seeds = sorted(set(k.split("_")[0] for k in data.keys()))
    n_seeds = len(seeds)

    fig, axes = plt.subplots(n_seeds, 2, figsize=(14, 3.5 * n_seeds), sharey="row")
    if n_seeds == 1:
        axes = axes[np.newaxis, :]

    for i, seed in enumerate(seeds):
        for j, mode in enumerate(["random", "seeded"]):
            key = f"{seed}_{mode}"
            v1 = np.array(data[key]["v1_total"])
            ax = axes[i, j]

            colors = []
            for th in THETAS:
                if is_cardinal(th):
                    colors.append("#d62728")  # red
                elif is_oblique(th):
                    colors.append("#1f77b4")  # blue
                else:
                    colors.append("#7f7f7f")  # gray

            bars = ax.bar(THETAS, v1, width=8, color=colors, edgecolor="k", linewidth=0.5)
            ax.axhline(v1.mean(), color="k", ls="--", lw=1, alpha=0.5,
                       label=f"mean={v1.mean():.1f}")

            # Cardinal/oblique means
            card_mask = np.array([is_cardinal(th) for th in THETAS])
            obliq_mask = np.array([is_oblique(th) for th in THETAS])
            card_mean = v1[card_mask].mean()
            obliq_mean = v1[obliq_mask].mean()
            ratio = card_mean / max(obliq_mean, 1e-12)

            ax.set_title(f"{seed}, {mode} init\n"
                         f"Card={card_mean:.1f}, Obliq={obliq_mean:.1f}, ratio={ratio:.3f}",
                         fontsize=10)
            ax.set_xlabel("Orientation (°)")
            if j == 0:
                ax.set_ylabel("V1 total spikes")
            ax.set_xticks(np.arange(0, 180, 30))
            ax.legend(fontsize=8)

    # Legend patch
    fig.text(0.5, 0.01,
             "Red = cardinal (0°/90° ± 10°)    Blue = oblique (45°/135° ± 10°)    Gray = other",
             ha="center", fontsize=10, style="italic")
    fig.suptitle("V1 spike counts per orientation (fresh network, no plasticity)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_v1_spikes_per_theta.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1_v1_spikes_per_theta.png")


def fig2_cardinal_oblique_ratio(data):
    """Bar chart comparing cardinal/oblique V1 ratio across seeds, random vs seeded."""
    seeds = sorted(set(k.split("_")[0] for k in data.keys()))

    random_ratios = []
    seeded_ratios = []
    for seed in seeds:
        for mode, lst in [("random", random_ratios), ("seeded", seeded_ratios)]:
            key = f"{seed}_{mode}"
            v1 = np.array(data[key]["v1_total"])
            card_mask = np.array([is_cardinal(th) for th in THETAS])
            obliq_mask = np.array([is_oblique(th) for th in THETAS])
            ratio = v1[card_mask].mean() / max(v1[obliq_mask].mean(), 1e-12)
            lst.append(ratio)

    random_ratios = np.array(random_ratios)
    seeded_ratios = np.array(seeded_ratios)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    x = np.arange(len(seeds))
    w = 0.35
    bars_r = ax.bar(x - w/2, random_ratios, w, label="Random init",
                    color="#d62728", alpha=0.7, edgecolor="k")
    bars_s = ax.bar(x + w/2, seeded_ratios, w, label="Seeded init (MF-norm)",
                    color="#1f77b4", alpha=0.7, edgecolor="k")

    # Reference line at 1.0
    ax.axhline(1.0, color="k", ls="-", lw=1.5, alpha=0.4)
    ax.axhspan(0.95, 1.05, color="green", alpha=0.08, label="±5% zone")

    # Annotate each bar
    for b in bars_r:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for b in bars_s:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("seed", "Seed ") for s in seeds])
    ax.set_ylabel("Cardinal / Oblique V1 spike ratio")
    ax.set_title("Cardinal bias: V1 spike ratio\n(fresh network, no plasticity)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0.85, max(max(random_ratios), max(seeded_ratios)) + 0.08)

    # Summary text
    ax.text(0.02, 0.02,
            f"Random: {random_ratios.mean():.3f} ± {random_ratios.std():.3f}\n"
            f"Seeded: {seeded_ratios.mean():.3f} ± {seeded_ratios.std():.3f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_cardinal_oblique_ratio.png"), dpi=200)
    plt.close(fig)
    print(f"  Saved fig2_cardinal_oblique_ratio.png")


def fig3_polar_v1_spikes(data):
    """Polar plot of V1 spikes per orientation, one panel per init mode, averaged across seeds."""
    seeds = sorted(set(k.split("_")[0] for k in data.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), subplot_kw=dict(projection="polar"))

    for j, mode in enumerate(["random", "seeded"]):
        ax = axes[j]
        all_v1 = []
        for seed in seeds:
            key = f"{seed}_{mode}"
            all_v1.append(np.array(data[key]["v1_total"]))
        all_v1 = np.array(all_v1)  # (n_seeds, K)
        mean_v1 = all_v1.mean(axis=0)
        std_v1 = all_v1.std(axis=0)

        # Convert orientation to radians (double for 180° → 360° mapping)
        theta_rad = np.deg2rad(THETAS * 2)
        # Close the curve
        theta_closed = np.concatenate([theta_rad, [theta_rad[0]]])
        mean_closed = np.concatenate([mean_v1, [mean_v1[0]]])
        std_closed = np.concatenate([std_v1, [std_v1[0]]])

        ax.plot(theta_closed, mean_closed, "k-", lw=2)
        ax.fill_between(theta_closed, mean_closed - std_closed,
                        mean_closed + std_closed, alpha=0.2, color="steelblue")

        # Mark cardinals and obliques
        for i, th in enumerate(THETAS):
            if is_cardinal(th):
                ax.plot(theta_rad[i], mean_v1[i], "ro", ms=8, zorder=5)
            elif is_oblique(th):
                ax.plot(theta_rad[i], mean_v1[i], "bs", ms=8, zorder=5)

        # Orientation labels
        ori_labels = [0, 45, 90, 135]
        ax.set_xticks([np.deg2rad(2 * o) for o in ori_labels])
        ax.set_xticklabels([f"{o}°" for o in ori_labels], fontsize=10)

        card_mask = np.array([is_cardinal(th) for th in THETAS])
        obliq_mask = np.array([is_oblique(th) for th in THETAS])
        ratio = mean_v1[card_mask].mean() / max(mean_v1[obliq_mask].mean(), 1e-12)

        ax.set_title(f"{mode.capitalize()} init\n"
                     f"Card/Obliq = {ratio:.3f}", pad=15, fontsize=11)

    fig.suptitle("V1 spike polar profile (mean ± std across seeds)\n"
                 "Red = cardinal, Blue = oblique",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_polar_v1_spikes.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3_polar_v1_spikes.png")


def fig4_lgn_vs_v1_bias(data):
    """Side-by-side: LGN ratio vs V1 ratio to show where bias amplification happens."""
    seeds = sorted(set(k.split("_")[0] for k in data.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for j, mode in enumerate(["random", "seeded"]):
        ax = axes[j]
        all_lgn = []
        all_v1 = []
        for seed in seeds:
            key = f"{seed}_{mode}"
            lgn = np.array(data[key]["lgn_total"])
            v1 = np.array(data[key]["v1_total"])
            all_lgn.append(lgn)
            all_v1.append(v1)
        all_lgn = np.array(all_lgn)
        all_v1 = np.array(all_v1)
        mean_lgn = all_lgn.mean(axis=0)
        mean_v1 = all_v1.mean(axis=0)

        # Normalize to mean=1 for comparison
        norm_lgn = mean_lgn / mean_lgn.mean()
        norm_v1 = mean_v1 / mean_v1.mean()

        colors_lgn = []
        colors_v1 = []
        for th in THETAS:
            if is_cardinal(th):
                colors_lgn.append("#d62728")
                colors_v1.append("#d62728")
            elif is_oblique(th):
                colors_lgn.append("#1f77b4")
                colors_v1.append("#1f77b4")
            else:
                colors_lgn.append("#7f7f7f")
                colors_v1.append("#7f7f7f")

        x = np.arange(len(THETAS))
        w = 0.38
        ax.bar(x - w/2, norm_lgn, w, color=colors_lgn, alpha=0.5, edgecolor="k",
               linewidth=0.3, label="LGN (normalized)")
        ax.bar(x + w/2, norm_v1, w, color=colors_v1, alpha=0.9, edgecolor="k",
               linewidth=0.3, label="V1 (normalized)")

        ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.4)
        ax.set_xticks(x[::3])
        ax.set_xticklabels([f"{int(th)}°" for th in THETAS[::3]], fontsize=9)
        ax.set_xlabel("Orientation")
        ax.set_ylabel("Normalized spike count (mean=1)")
        ax.set_ylim(0.8, 1.25)

        card_mask = np.array([is_cardinal(th) for th in THETAS])
        obliq_mask = np.array([is_oblique(th) for th in THETAS])
        lgn_ratio = mean_lgn[card_mask].mean() / max(mean_lgn[obliq_mask].mean(), 1e-12)
        v1_ratio = mean_v1[card_mask].mean() / max(mean_v1[obliq_mask].mean(), 1e-12)

        ax.set_title(f"{mode.capitalize()} init\n"
                     f"LGN ratio={lgn_ratio:.3f}, V1 ratio={v1_ratio:.3f}",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("LGN vs V1 spike counts per orientation (normalized to mean)\n"
                 "Light bars = LGN, Dark bars = V1 | Red = cardinal, Blue = oblique",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_lgn_vs_v1_bias.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4_lgn_vs_v1_bias.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading data from {DATA_PATH}")
    data = load_data()
    print(f"Creating visualizations in {OUT_DIR}")

    fig1_spikes_per_theta(data)
    fig2_cardinal_oblique_ratio(data)
    fig3_polar_v1_spikes(data)
    fig4_lgn_vs_v1_bias(data)

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnostic: Measure cardinal vs oblique orientation bias at each processing stage.

This script creates fresh networks (random init, seeded init) with NO plasticity
and measures spike counts + drive strength at each level:
  1. LGN spikes per orientation
  2. Total feedforward current (I_ff) per orientation per ensemble
  3. V1 spikes per orientation per ensemble

The goal is to localize WHERE cardinal bias originates:
  - If LGN spikes are biased → the grating sampling / DoG / spiking mechanism
  - If LGN spikes are unbiased but V1 spikes are biased → threshold nonlinearity
  - If neither is biased on a fresh network but bias emerges after STDP → learning amplification

Usage:
    python -u diag_cardinal_bias.py [--out DIR] [--seeds N]
"""
import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict

from biologically_plausible_v1_stdp import (
    Params, RgcLgnV1Network, compute_osi,
)


def measure_spike_counts_per_theta(
    net: RgcLgnV1Network,
    thetas_deg: np.ndarray,
    repeats: int = 10,
    contrast: float = 1.0,
) -> dict:
    """Run gratings at each orientation, collect LGN + V1 spike counts (no plasticity).

    Returns dict with:
        lgn_total[j]  : total LGN spikes summed over all neurons, shape (K,)
        v1_total[j]   : total V1 spikes summed over all ensembles, shape (K,)
        v1_per_ens[j]  : V1 spikes per ensemble, shape (K, M)
        lgn_per_ens[j] : LGN spikes per ON/OFF, shape (K, 2) [ON total, OFF total]
    """
    p = net.p
    K = len(thetas_deg)
    M = p.M
    N2 = p.N * p.N

    lgn_total = np.zeros(K, dtype=np.float64)
    v1_total = np.zeros(K, dtype=np.float64)
    v1_per_ens = np.zeros((K, M), dtype=np.float64)
    lgn_on_total = np.zeros(K, dtype=np.float64)
    lgn_off_total = np.zeros(K, dtype=np.float64)

    snap = net.save_dynamic_state()

    for j, th in enumerate(thetas_deg):
        for _ in range(repeats):
            net.reset_state()
            counts = net.run_segment_counts(float(th), plastic=False, contrast=contrast)
            lgn_total[j] += float(counts["lgn_counts"].sum())
            lgn_on_total[j] += float(counts["lgn_counts"][:N2].sum())
            lgn_off_total[j] += float(counts["lgn_counts"][N2:].sum())
            v1_total[j] += float(counts["v1_counts"].sum())
            v1_per_ens[j] += counts["v1_counts"].astype(np.float64)

    net.restore_dynamic_state(snap)

    # Average over repeats
    lgn_total /= repeats
    lgn_on_total /= repeats
    lgn_off_total /= repeats
    v1_total /= repeats
    v1_per_ens /= repeats

    return {
        "lgn_total": lgn_total,
        "lgn_on": lgn_on_total,
        "lgn_off": lgn_off_total,
        "v1_total": v1_total,
        "v1_per_ens": v1_per_ens,
    }


def classify_orientations(thetas_deg: np.ndarray, tol_deg: float = 10.0):
    """Classify orientations as cardinal (near 0° or 90°) or oblique (near 45° or 135°)."""
    cardinal_mask = np.zeros(len(thetas_deg), dtype=bool)
    oblique_mask = np.zeros(len(thetas_deg), dtype=bool)
    for i, th in enumerate(thetas_deg):
        # Distance to nearest cardinal (0° or 90° mod 180°)
        d_card = min(th % 180, abs(th % 180 - 90), abs(th % 180 - 180))
        # Distance to nearest oblique (45° or 135° mod 180°)
        d_obliq = min(abs(th % 180 - 45), abs(th % 180 - 135))
        if d_card <= tol_deg:
            cardinal_mask[i] = True
        elif d_obliq <= tol_deg:
            oblique_mask[i] = True
    return cardinal_mask, oblique_mask


def report_bias(label: str, values: np.ndarray, thetas_deg: np.ndarray):
    """Print cardinal/oblique ratio for a 1D array of per-theta values."""
    card_mask, obliq_mask = classify_orientations(thetas_deg)
    card_mean = float(values[card_mask].mean()) if card_mask.any() else 0.0
    obliq_mean = float(values[obliq_mask].mean()) if obliq_mask.any() else 0.0
    ratio = card_mean / max(obliq_mean, 1e-12)
    print(f"  {label}:")
    print(f"    Cardinal mean = {card_mean:.2f}, Oblique mean = {obliq_mean:.2f}, "
          f"ratio = {ratio:.4f}")
    # Also print per-theta values
    for i, th in enumerate(thetas_deg):
        tag = ""
        if card_mask[i]:
            tag = " [CARDINAL]"
        elif obliq_mask[i]:
            tag = " [OBLIQUE]"
        print(f"    θ={th:6.1f}°: {values[i]:8.2f}{tag}")
    return ratio


def run_diagnostic(p: Params, init_mode: str, thetas_deg: np.ndarray,
                   repeats: int, label: str):
    """Run one full diagnostic for a given init mode."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  init_mode={init_mode}, N={p.N}, M={p.M}, seed={p.seed}")
    print(f"  repeats={repeats}, segment_ms={p.segment_ms}")
    print(f"{'='*70}")

    net = RgcLgnV1Network(p, init_mode=init_mode)
    data = measure_spike_counts_per_theta(net, thetas_deg, repeats=repeats)

    print("\n--- LGN total spikes per orientation ---")
    lgn_ratio = report_bias("LGN total", data["lgn_total"], thetas_deg)

    print("\n--- LGN ON spikes per orientation ---")
    report_bias("LGN ON", data["lgn_on"], thetas_deg)

    print("\n--- LGN OFF spikes per orientation ---")
    report_bias("LGN OFF", data["lgn_off"], thetas_deg)

    print("\n--- V1 total spikes per orientation (summed over all ensembles) ---")
    v1_ratio = report_bias("V1 total", data["v1_total"], thetas_deg)

    # Per-ensemble breakdown: which ensembles spike most at cardinal vs oblique?
    v1_per_ens = data["v1_per_ens"]  # (K, M)
    card_mask, obliq_mask = classify_orientations(thetas_deg)
    ens_card = v1_per_ens[card_mask].mean(axis=0) if card_mask.any() else np.zeros(p.M)
    ens_obliq = v1_per_ens[obliq_mask].mean(axis=0) if obliq_mask.any() else np.zeros(p.M)
    print(f"\n--- V1 per-ensemble: cardinal vs oblique mean spikes ---")
    print(f"  {'Ens':>4s}  {'Cardinal':>10s}  {'Oblique':>10s}  {'Ratio':>8s}")
    for m in range(p.M):
        r = ens_card[m] / max(ens_obliq[m], 1e-12)
        print(f"  {m:4d}  {ens_card[m]:10.2f}  {ens_obliq[m]:10.2f}  {r:8.3f}")

    # Also evaluate OSI/pref on this fresh (unlearned) network
    rates = net.evaluate_tuning(thetas_deg, repeats=repeats, contrast=1.0)
    osi, pref = compute_osi(rates, thetas_deg)
    tuned = osi >= 0.3
    print(f"\n--- Fresh network tuning (no plasticity) ---")
    print(f"  Mean OSI = {float(osi.mean()):.3f}, tuned (OSI>=0.3) = {int(tuned.sum())}/{p.M}")
    if tuned.any():
        prefs_tuned = pref[tuned]
        card_pref = sum(1 for pr in prefs_tuned if min(pr % 180, abs(pr % 180 - 90), abs(pr % 180 - 180)) < 15)
        obliq_pref = sum(1 for pr in prefs_tuned if min(abs(pr % 180 - 45), abs(pr % 180 - 135)) < 15)
        other_pref = int(tuned.sum()) - card_pref - obliq_pref
        print(f"  Preferred cardinal (within 15° of 0°/90°): {card_pref}")
        print(f"  Preferred oblique (within 15° of 45°/135°): {obliq_pref}")
        print(f"  Preferred other: {other_pref}")

    return {
        "lgn_ratio": lgn_ratio,
        "v1_ratio": v1_ratio,
        "lgn_total": data["lgn_total"].tolist(),
        "v1_total": data["v1_total"].tolist(),
        "osi_mean": float(osi.mean()),
        "n_tuned": int(tuned.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Cardinal bias diagnostic")
    parser.add_argument("--out", default="/tmp/diag_cardinal", help="Output directory")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds to test")
    parser.add_argument("--repeats", type=int, default=10, help="Repeats per orientation")
    parser.add_argument("--M", type=int, default=32, help="Number of V1 ensembles")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Test orientations: dense sampling including cardinal and oblique
    thetas_deg = np.arange(0.0, 180.0, 10.0)

    all_results = {}

    for seed in range(1, args.seeds + 1):
        # --- Random init ---
        p_rand = Params(N=8, M=args.M, seed=seed, v1_bias_eta=0.0)
        res_rand = run_diagnostic(
            p_rand, "random", thetas_deg, args.repeats,
            f"Seed {seed}: RANDOM init (M={args.M})"
        )
        all_results[f"seed{seed}_random"] = res_rand

        # --- Seeded init ---
        p_seed = Params(N=8, M=args.M, seed=seed, v1_bias_eta=0.0)
        res_seed = run_diagnostic(
            p_seed, "seeded", thetas_deg, args.repeats,
            f"Seed {seed}: SEEDED init (M={args.M})"
        )
        all_results[f"seed{seed}_seeded"] = res_seed

    # --- Aggregate summary ---
    print(f"\n{'='*70}")
    print("  AGGREGATE SUMMARY")
    print(f"{'='*70}")

    for init_mode in ["random", "seeded"]:
        lgn_ratios = []
        v1_ratios = []
        for seed in range(1, args.seeds + 1):
            key = f"seed{seed}_{init_mode}"
            lgn_ratios.append(all_results[key]["lgn_ratio"])
            v1_ratios.append(all_results[key]["v1_ratio"])
        lgn_ratios = np.array(lgn_ratios)
        v1_ratios = np.array(v1_ratios)
        print(f"\n  init_mode={init_mode} (n={args.seeds} seeds):")
        print(f"    LGN cardinal/oblique ratio: {lgn_ratios.mean():.4f} ± {lgn_ratios.std():.4f}")
        print(f"    V1  cardinal/oblique ratio: {v1_ratios.mean():.4f} ± {v1_ratios.std():.4f}")

    # Save results
    outpath = os.path.join(args.out, "cardinal_bias_results.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {outpath}")


if __name__ == "__main__":
    main()

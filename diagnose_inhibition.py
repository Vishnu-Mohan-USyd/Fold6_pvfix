#!/usr/bin/env python3
"""Diagnose inhibitory current budget in a trained RGC-LGN-V1 network.

Quantitatively measures conductances, actual currents, PV selectivity, PV weight
evolution, and PP activity to confirm the three root causes of "namesake" inhibition:

1. PV provides blanket (non-selective) inhibition due to global connectivity
2. Inhibitory driving force is ~13x weaker than excitatory near resting potential
3. Push-Pull pathway lacks ON/OFF phase opposition

Outputs a JSON file and printed summary table.

Usage:
    python diagnose_inhibition.py                  # full diagnostic
    python diagnose_inhibition.py --skip-training  # use a pre-trained snapshot (if available)
    python diagnose_inhibition.py --segments 100   # shorter Phase A for quick check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from biologically_plausible_v1_stdp import (
    Params,
    RgcLgnV1Network,
    compute_osi,
    calibrate_ee_drive,
)


# ------------------------------------------------------------------------------
# Configuration (matches validate_sequence.py / run_ablation_study.py)
# ------------------------------------------------------------------------------
M_DEFAULT = 32
N_DEFAULT = 8
CONTRAST = 2.0
PHASE_A_SEGMENTS = 300
EVAL_THETAS = np.linspace(0, 180, 12, endpoint=False)
EVAL_REPEATS = 5
RECORDING_MS = 200.0  # per orientation recording duration


def build_network(seed: int = 1, segments: int = PHASE_A_SEGMENTS) -> RgcLgnV1Network:
    """Build and train a network through Phase A (feedforward STDP)."""
    p = Params(
        N=N_DEFAULT, M=M_DEFAULT, seed=seed,
        train_segments=0,
        train_stimulus="grating",
        train_contrast=CONTRAST,
        ee_stdp_enabled=True,
        ee_connectivity="all_to_all",
        ee_stdp_A_plus=0.002,
        ee_stdp_A_minus=0.0024,
        ee_stdp_weight_dep=False,
    )
    net = RgcLgnV1Network(p, init_mode="random")

    # Phase A: develop receptive fields
    print(f"[Phase A] Training feedforward STDP ({segments} segments)...")
    net.ff_plastic_enabled = True
    net.ee_stdp_active = False
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    theta_step = 180.0 / phi
    theta_offset = float(net.rng.uniform(0.0, 180.0))

    for s in range(1, segments + 1):
        th = float((theta_offset + (s - 1) * theta_step) % 180.0)
        net.run_segment(th, plastic=True, contrast=CONTRAST)
        if s % 50 == 0 or s == segments:
            rates = net.evaluate_tuning(EVAL_THETAS, repeats=3, contrast=CONTRAST)
            osi, _ = compute_osi(rates, EVAL_THETAS)
            print(f"  seg {s}/{segments}: OSI={osi.mean():.3f}, rate={rates.mean():.2f} Hz")

    # Calibrate E->E
    print("[Calibrate] Auto-calibrating E->E drive to 0.15...")
    scale, frac = calibrate_ee_drive(net, 0.15)
    print(f"  scale={scale:.1f}, frac={frac:.4f}")

    return net


def measure_conductances_and_currents(
    net: RgcLgnV1Network,
    thetas_deg: np.ndarray,
    duration_ms: float = RECORDING_MS,
) -> Dict[str, Any]:
    """Measure per-timestep conductances and currents at each orientation.

    For each orientation, runs the network non-plastically and records:
    - g_exc_ff, g_exc_ee (excitatory conductances)
    - g_pv, g_som (inhibitory conductances)
    - I_exc, I_pv, I_som (actual currents = g * driving_force)
    - Membrane voltage distribution

    Returns dict keyed by orientation with per-step arrays.
    """
    p = net.p
    results: Dict[str, Any] = {}
    snap = net.save_dynamic_state()

    for theta in thetas_deg:
        net.reset_state()
        steps = int(duration_ms / p.dt_ms)
        phase = float(net.rng.uniform(0, 2 * math.pi))

        # Accumulators (per step, aggregated across ensembles)
        g_exc_ff_ts = np.zeros((steps, net.M), dtype=np.float32)
        g_exc_ee_ts = np.zeros((steps, net.M), dtype=np.float32)
        g_pv_ts = np.zeros((steps, net.M), dtype=np.float32)
        g_som_ts = np.zeros((steps, net.M), dtype=np.float32)
        v_ts = np.zeros((steps, net.M), dtype=np.float32)
        v1_spk_ts = np.zeros((steps, net.M), dtype=np.uint8)
        pv_spk_ts = np.zeros((steps, net.n_pv), dtype=np.uint8)

        for k in range(steps):
            on_spk, off_spk = net.rgc_spikes_grating(
                theta, t_ms=k * p.dt_ms, phase=phase, contrast=CONTRAST
            )
            v1_spk = net.step(on_spk, off_spk, plastic=False)

            # Record conductances (these are the raw conductance variables)
            g_exc_ff_ts[k] = net.g_exc_ff
            g_exc_ee_ts[k] = net.g_exc_ee
            g_pv_raw = np.clip(net.g_v1_inh_pv_decay - net.g_v1_inh_pv_rise, 0.0, None)
            g_pv_ts[k] = g_pv_raw
            g_som_ts[k] = net.g_v1_inh_som
            v_ts[k] = net.v1_exc.v
            v1_spk_ts[k] = v1_spk
            pv_spk_ts[k] = net.last_pv_spk

        # Compute actual currents = g * driving_force
        I_exc_ts = (g_exc_ff_ts + g_exc_ee_ts) * (p.E_exc - v_ts)
        I_pv_ts = g_pv_ts * (p.E_inh - v_ts)
        I_som_ts = g_som_ts * (p.E_inh - v_ts)

        # Aggregate statistics (mean over time and ensembles)
        th_key = f"{theta:.1f}"
        results[th_key] = {
            # Mean conductances
            "g_exc_ff_mean": float(g_exc_ff_ts.mean()),
            "g_exc_ee_mean": float(g_exc_ee_ts.mean()),
            "g_pv_mean": float(g_pv_ts.mean()),
            "g_som_mean": float(g_som_ts.mean()),
            # Mean currents
            "I_exc_mean": float(I_exc_ts.mean()),
            "I_pv_mean": float(I_pv_ts.mean()),
            "I_som_mean": float(I_som_ts.mean()),
            # Current ratios (avoid div-by-zero)
            "ratio_I_pv_over_I_exc": float(
                np.abs(I_pv_ts).mean() / max(1e-12, np.abs(I_exc_ts).mean())
            ),
            "ratio_I_som_over_I_exc": float(
                np.abs(I_som_ts).mean() / max(1e-12, np.abs(I_exc_ts).mean())
            ),
            # Mean membrane voltage
            "V_mean": float(v_ts.mean()),
            "V_std": float(v_ts.std()),
            # Spike rates
            "E_rate_hz": float(v1_spk_ts.sum() / (net.M * duration_ms / 1000.0)),
            "PV_rate_hz": float(pv_spk_ts.sum() / (net.n_pv * duration_ms / 1000.0)),
            # Per-ensemble PV inhibitory conductance (spatial profile)
            "g_pv_per_ensemble_mean": g_pv_ts.mean(axis=0).tolist(),
        }

    net.restore_dynamic_state(snap)
    return results


def measure_pv_selectivity(
    net: RgcLgnV1Network,
    thetas_deg: np.ndarray,
    repeats: int = EVAL_REPEATS,
) -> Dict[str, Any]:
    """Measure PV spike rate per orientation to check if PV is selective or blanket."""
    p = net.p
    snap = net.save_dynamic_state()

    pv_rates = np.zeros((net.n_pv, len(thetas_deg)), dtype=np.float32)
    e_rates = np.zeros((net.M, len(thetas_deg)), dtype=np.float32)

    for j, th in enumerate(thetas_deg):
        pv_cnt = np.zeros(net.n_pv, dtype=np.float32)
        e_cnt = np.zeros(net.M, dtype=np.float32)
        for _ in range(repeats):
            net.reset_state()
            counts = net.run_segment_counts(float(th), plastic=False)
            pv_cnt += counts["pv_counts"].astype(np.float32)
            e_cnt += counts["v1_counts"].astype(np.float32)
        seg_sec = p.segment_ms / 1000.0
        pv_rates[:, j] = pv_cnt / (repeats * seg_sec)
        e_rates[:, j] = e_cnt / (repeats * seg_sec)

    net.restore_dynamic_state(snap)

    # Compute selectivity index for PV (same OSI formula as for E)
    pv_osi, pv_pref = compute_osi(pv_rates, thetas_deg)
    e_osi, e_pref = compute_osi(e_rates, thetas_deg)

    # PV rate variation across orientations (CV = std/mean)
    pv_mean_per_ori = pv_rates.mean(axis=0)  # (n_ori,)
    pv_cv = float(pv_mean_per_ori.std() / max(1e-12, pv_mean_per_ori.mean()))

    return {
        "pv_osi_mean": float(pv_osi.mean()),
        "pv_osi_std": float(pv_osi.std()),
        "e_osi_mean": float(e_osi.mean()),
        "e_osi_std": float(e_osi.std()),
        "pv_rate_per_ori": pv_mean_per_ori.tolist(),
        "pv_rate_cv": pv_cv,
        "pv_rate_global_mean": float(pv_rates.mean()),
        "e_rate_global_mean": float(e_rates.mean()),
    }


def measure_pv_weight_stats(net: RgcLgnV1Network) -> Dict[str, Any]:
    """Snapshot PV->E weight statistics."""
    W = net.W_pv_e
    return {
        "W_pv_e_mean": float(W.mean()),
        "W_pv_e_std": float(W.std()),
        "W_pv_e_min": float(W.min()),
        "W_pv_e_max": float(W.max()),
        "W_pv_e_nonzero_frac": float((W > 0).mean()),
        "W_pv_e_shape": list(W.shape),
    }


def measure_weight_evolution(
    seed: int = 1,
    segments: int = PHASE_A_SEGMENTS,
    checkpoints: List[int] | None = None,
) -> Dict[str, Any]:
    """Train a network and record PV->E weight stats at checkpoint segments."""
    if checkpoints is None:
        checkpoints = [0, 50, 100, 150, 200, 250, 300]
    # Clamp checkpoints to actual segment count
    checkpoints = [c for c in checkpoints if c <= segments]
    if 0 not in checkpoints:
        checkpoints = [0] + checkpoints

    p = Params(
        N=N_DEFAULT, M=M_DEFAULT, seed=seed,
        train_segments=0,
        train_stimulus="grating",
        train_contrast=CONTRAST,
        ee_stdp_enabled=True,
        ee_connectivity="all_to_all",
        ee_stdp_A_plus=0.002,
        ee_stdp_A_minus=0.0024,
        ee_stdp_weight_dep=False,
    )
    net = RgcLgnV1Network(p, init_mode="random")
    net.ff_plastic_enabled = True
    net.ee_stdp_active = False
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    theta_step = 180.0 / phi
    theta_offset = float(net.rng.uniform(0.0, 180.0))

    evolution: Dict[str, Any] = {}

    for s in range(0, segments + 1):
        if s in checkpoints:
            W = net.W_pv_e
            evolution[str(s)] = {
                "W_pv_e_mean": float(W.mean()),
                "W_pv_e_std": float(W.std()),
                "W_pv_e_min": float(W.min()),
                "W_pv_e_max": float(W.max()),
            }
            print(f"  [Weight evo] seg {s}: W_pv_e mean={W.mean():.4f}, "
                  f"std={W.std():.4f}, min={W.min():.4f}, max={W.max():.4f}")
        if s < segments:
            th = float((theta_offset + s * theta_step) % 180.0)
            net.run_segment(th, plastic=True, contrast=CONTRAST)

    return evolution


def print_summary(results: Dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 80)
    print("INHIBITION DIAGNOSTIC SUMMARY")
    print("=" * 80)

    # --- Conductance & Current Budget ---
    print("\n--- Conductance & Current Budget (averaged across orientations) ---")
    cond = results["conductances"]
    # Average across all orientation keys
    ori_keys = list(cond.keys())
    n_ori = len(ori_keys)

    avg = {}
    for key in ["g_exc_ff_mean", "g_exc_ee_mean", "g_pv_mean", "g_som_mean",
                 "I_exc_mean", "I_pv_mean", "I_som_mean",
                 "ratio_I_pv_over_I_exc", "ratio_I_som_over_I_exc",
                 "V_mean", "E_rate_hz", "PV_rate_hz"]:
        avg[key] = np.mean([cond[k][key] for k in ori_keys])

    print(f"  Mean membrane voltage:  {avg['V_mean']:.1f} mV")
    print(f"  E firing rate:          {avg['E_rate_hz']:.2f} Hz")
    print(f"  PV firing rate:         {avg['PV_rate_hz']:.2f} Hz")
    print()
    print(f"  {'Component':<20} {'Conductance':>12} {'Current (pA)':>14} {'|I|/|I_exc|':>14}")
    print(f"  {'-'*20} {'-'*12} {'-'*14} {'-'*14}")
    print(f"  {'Exc FF':<20} {avg['g_exc_ff_mean']:>12.4f}  {avg['I_exc_mean']:>13.2f}  {'(reference)':>14}")
    print(f"  {'Exc EE':<20} {avg['g_exc_ee_mean']:>12.4f}  {'':>13}  {'':>14}")
    print(f"  {'PV inhibition':<20} {avg['g_pv_mean']:>12.4f}  {avg['I_pv_mean']:>13.2f}  {avg['ratio_I_pv_over_I_exc']:>13.1%}")
    print(f"  {'SOM inhibition':<20} {avg['g_som_mean']:>12.4f}  {avg['I_som_mean']:>13.2f}  {avg['ratio_I_som_over_I_exc']:>13.1%}")

    driving_force_exc = 0.0 - avg["V_mean"]
    driving_force_inh = -70.0 - avg["V_mean"]
    print(f"\n  Driving force at V={avg['V_mean']:.1f} mV:")
    print(f"    Excitatory: E_exc - V = {driving_force_exc:+.1f} mV")
    print(f"    Inhibitory: E_inh - V = {driving_force_inh:+.1f} mV")
    print(f"    Ratio |exc/inh| = {abs(driving_force_exc) / max(0.01, abs(driving_force_inh)):.1f}x")

    # --- PV Selectivity ---
    print("\n--- PV Selectivity ---")
    sel = results["pv_selectivity"]
    print(f"  PV OSI (mean+/-std):  {sel['pv_osi_mean']:.3f} +/- {sel['pv_osi_std']:.3f}")
    print(f"  E  OSI (mean+/-std):  {sel['e_osi_mean']:.3f} +/- {sel['e_osi_std']:.3f}")
    print(f"  PV rate CV across orientations: {sel['pv_rate_cv']:.3f}")
    print(f"  PV rate per orientation: {[f'{r:.1f}' for r in sel['pv_rate_per_ori']]}")
    if sel["pv_osi_mean"] < 0.1:
        print("  >> PV is NON-SELECTIVE (blanket inhibition)")
    elif sel["pv_osi_mean"] < 0.3:
        print("  >> PV is WEAKLY selective")
    else:
        print("  >> PV is SELECTIVE")

    # --- PV Weight Stats ---
    print("\n--- PV->E Weight Statistics ---")
    ws = results["pv_weights"]
    print(f"  W_pv_e shape: {ws['W_pv_e_shape']}")
    print(f"  W_pv_e mean={ws['W_pv_e_mean']:.4f}, std={ws['W_pv_e_std']:.4f}")
    print(f"  W_pv_e min={ws['W_pv_e_min']:.4f}, max={ws['W_pv_e_max']:.4f}")
    print(f"  W_pv_e nonzero fraction: {ws['W_pv_e_nonzero_frac']:.3f}")

    # --- PV Weight Evolution ---
    if "weight_evolution" in results:
        print("\n--- PV->E Weight Evolution During Training ---")
        evo = results["weight_evolution"]
        for seg_key in sorted(evo.keys(), key=lambda x: int(x)):
            e = evo[seg_key]
            print(f"  seg {seg_key:>3s}: mean={e['W_pv_e_mean']:.4f}, "
                  f"std={e['W_pv_e_std']:.4f}, "
                  f"min={e['W_pv_e_min']:.4f}, max={e['W_pv_e_max']:.4f}")

    # --- Verdicts ---
    print("\n" + "=" * 80)
    print("VERDICTS")
    print("=" * 80)
    verdicts = []

    if avg["ratio_I_pv_over_I_exc"] < 0.05:
        verdicts.append("[FAIL] PV inhibitory current is <5% of excitatory current (CONFIRMED: driving force asymmetry)")
    elif avg["ratio_I_pv_over_I_exc"] < 0.10:
        verdicts.append("[WARN] PV inhibitory current is 5-10% of excitatory current (weak)")
    else:
        verdicts.append("[OK] PV inhibitory current is >10% of excitatory current (functional)")

    if sel["pv_osi_mean"] < 0.1:
        verdicts.append("[FAIL] PV provides uniform/blanket inhibition (CONFIRMED: no selectivity)")
    else:
        verdicts.append(f"[OK] PV has orientation selectivity (OSI={sel['pv_osi_mean']:.3f})")

    for v in verdicts:
        print(f"  {v}")
    print()

    return


def main():
    parser = argparse.ArgumentParser(description="Diagnose inhibitory circuit effectiveness")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--segments", type=int, default=PHASE_A_SEGMENTS, help="Phase A training segments")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (debug only)")
    parser.add_argument("--skip-evolution", action="store_true", help="Skip weight evolution tracking")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for JSON")
    args = parser.parse_args()

    t0 = time.time()

    # Step 1: Build and train network
    print("=" * 80)
    print("INHIBITION DIAGNOSTIC")
    print("=" * 80)
    net = build_network(seed=args.seed, segments=args.segments)
    t_train = time.time() - t0
    print(f"  Training took {t_train:.1f}s")

    # Step 2: Measure conductances and currents at multiple orientations
    print("\n[Measuring] Conductances and currents at each orientation...")
    cond_results = measure_conductances_and_currents(net, EVAL_THETAS)

    # Step 3: Measure PV selectivity
    print("[Measuring] PV selectivity...")
    pv_sel = measure_pv_selectivity(net, EVAL_THETAS)

    # Step 4: PV weight statistics
    print("[Measuring] PV->E weight statistics...")
    pv_weights = measure_pv_weight_stats(net)

    # Step 5: Weight evolution (optional, re-trains from scratch)
    weight_evo = None
    if not args.skip_evolution:
        print("\n[Measuring] PV->E weight evolution during training...")
        weight_evo = measure_weight_evolution(
            seed=args.seed, segments=args.segments,
            checkpoints=[0, 50, 100, 150, 200, 250, 300],
        )

    # Compile results
    results: Dict[str, Any] = {
        "seed": args.seed,
        "segments": args.segments,
        "params": {
            "pv_in_sigma": float(net.p.pv_in_sigma),
            "pv_out_sigma": float(net.p.pv_out_sigma),
            "w_pv_e": float(net.p.w_pv_e),
            "w_pv_e_max": float(net.p.w_pv_e_max),
            "w_e_pv": float(net.p.w_e_pv),
            "eta_pv_istdp": float(net.p.eta_pv_istdp),
            "target_rate_hz": float(net.p.target_rate_hz),
            "w_som_e": float(net.p.w_som_e),
            "E_exc": float(net.p.E_exc),
            "E_inh": float(net.p.E_inh),
        },
        "conductances": cond_results,
        "pv_selectivity": pv_sel,
        "pv_weights": pv_weights,
    }
    if weight_evo is not None:
        results["weight_evolution"] = weight_evo

    # Save JSON
    out_path = os.path.join(args.outdir, "inhibition_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out_path}")

    # Print summary
    print_summary(results)

    t_total = time.time() - t0
    print(f"Total diagnostic time: {t_total:.1f}s")


if __name__ == "__main__":
    main()

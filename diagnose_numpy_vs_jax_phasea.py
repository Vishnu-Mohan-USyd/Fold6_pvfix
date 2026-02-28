#!/usr/bin/env python
"""Compare pref orientation distribution: JAX Phase A vs numpy Phase A with 300 segments.

Hypothesis: JAX Phase A produces clustered prefs, numpy produces better-distributed ones.
"""

import sys
import time
import math
import numpy as np

sys.path.insert(0, '.')

from biologically_plausible_v1_stdp import (
    Params, RgcLgnV1Network, compute_osi, calibrate_ee_drive,
)
from network_jax import (
    numpy_net_to_jax_state,
    run_segment_jax, run_sequence_trial_jax, reset_state_jax,
    evaluate_tuning_jax,
)
import jax

GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
THETA_STEP = 180.0 / GOLDEN_RATIO
SEQ_THETAS = [0.0, 45.0, 90.0, 135.0]
ELEMENT_MS = 150.0
ITI_MS = 1500.0
CONTRAST = 1.0


def compute_fwd_rev_ratio(W_e_e, pref, seq_thetas):
    fwd_ws, rev_ws = [], []
    for ei in range(len(seq_thetas) - 1):
        pre_th, post_th = seq_thetas[ei], seq_thetas[ei + 1]
        d_pre = np.abs(pref - pre_th)
        d_pre = np.minimum(d_pre, 180.0 - d_pre)
        d_post = np.abs(pref - post_th)
        d_post = np.minimum(d_post, 180.0 - d_post)
        pre_mask = d_pre < 22.5
        post_mask = d_post < 22.5
        for pi in np.where(post_mask)[0]:
            for pj in np.where(pre_mask)[0]:
                if pi != pj:
                    fwd_ws.append(W_e_e[pi, pj])
                    rev_ws.append(W_e_e[pj, pi])
    if len(fwd_ws) == 0:
        return 0.0, 0.0, 1.0, 0
    fwd_m = float(np.mean(fwd_ws))
    rev_m = float(np.mean(rev_ws))
    ratio = fwd_m / max(1e-10, rev_m)
    return fwd_m, rev_m, ratio, len(fwd_ws)


def analyze_pref_coverage(pref, label):
    """Analyze neuron coverage for each sequence element."""
    print(f"\n  {label}:")
    print(f"    Sorted prefs: {np.sort(pref).round(1)}")
    total_pairs = 0
    for th in SEQ_THETAS:
        d = np.abs(pref - th)
        d = np.minimum(d, 180 - d)
        n = np.sum(d < 22.5)
        print(f"      Theta {th:5.1f}°: {n} neurons")
    # Count pairs
    for ei in range(len(SEQ_THETAS) - 1):
        pre_th, post_th = SEQ_THETAS[ei], SEQ_THETAS[ei + 1]
        d_pre = np.abs(pref - pre_th)
        d_pre = np.minimum(d_pre, 180 - d_pre)
        d_post = np.abs(pref - post_th)
        d_post = np.minimum(d_post, 180 - d_post)
        n_pre = np.sum(d_pre < 22.5)
        n_post = np.sum(d_post < 22.5)
        n_pairs_transition = n_pre * n_post
        total_pairs += n_pairs_transition
        print(f"      {SEQ_THETAS[ei]}→{post_th}: {n_pre}×{n_post}={n_pairs_transition} pairs")
    print(f"    Total transition pairs: {total_pairs}")
    return total_pairs


def run_test(phase_a_mode, n_segments=300, seed=42, n_pres=400):
    """Run Phase A (JAX or numpy), calibrate, run Phase B, report F>R."""
    print(f"\n{'='*70}")
    print(f"Phase A: {phase_a_mode}, {n_segments} segments, seed={seed}")
    print(f"{'='*70}")

    p = Params(
        M=16, N=8, seed=seed,
        ee_stdp_enabled=True, ee_connectivity="all_to_all",
        ee_stdp_A_plus=0.005, ee_stdp_A_minus=0.006,
        ee_stdp_weight_dep=True, train_segments=0, segment_ms=300.0,
    )
    net = RgcLgnV1Network(p)

    thetas_eval = np.linspace(0, 180, 12, endpoint=False)

    if phase_a_mode == 'numpy':
        t0 = time.perf_counter()
        for seg in range(n_segments):
            theta = (seg * THETA_STEP) % 180.0
            net.run_segment(theta, plastic=True)
            if (seg + 1) % 100 == 0:
                print(f"    Phase A numpy: {seg+1}/{n_segments}")
        print(f"    Done in {time.perf_counter()-t0:.1f}s")

        # Evaluate tuning in numpy
        rates = net.evaluate_tuning(thetas_eval, repeats=2)
        osi_vals, pref_pre_cal = compute_osi(rates, thetas_eval)
        print(f"    Mean OSI (pre-calibration): {osi_vals.mean():.3f}")
        analyze_pref_coverage(pref_pre_cal, "Pre-calibration prefs (numpy)")

        # Calibrate
        scale, frac = calibrate_ee_drive(net, target_frac=0.15)
        mask_ee = net.mask_e_e.astype(bool) if hasattr(net, 'mask_e_e') else np.ones((16,16), dtype=bool)
        np.fill_diagonal(mask_ee, False)
        cal_mean = float(net.W_e_e[mask_ee].mean())
        new_w_max = max(cal_mean * 3.0, net.p.w_e_e_max)
        net.p.w_e_e_max = new_w_max
        print(f"    Calibration: scale={scale:.1f}, cal_mean={cal_mean:.4f}, w_max={new_w_max:.4f}")

        # Convert to JAX
        state, static = numpy_net_to_jax_state(net)

    elif phase_a_mode == 'jax':
        state, static = numpy_net_to_jax_state(net)
        t0 = time.perf_counter()
        for seg in range(n_segments):
            theta = (seg * THETA_STEP) % 180.0
            state, _ = run_segment_jax(state, static, theta, 1.0, True)
            if (seg + 1) % 100 == 0:
                print(f"    Phase A JAX: {seg+1}/{n_segments}")
        print(f"    Done in {time.perf_counter()-t0:.1f}s")

        # Evaluate tuning
        rates = evaluate_tuning_jax(state, static, thetas_eval, repeats=2)
        osi_vals, pref_pre_cal = compute_osi(rates, thetas_eval)
        print(f"    Mean OSI (pre-calibration): {osi_vals.mean():.3f}")
        analyze_pref_coverage(pref_pre_cal, "Pre-calibration prefs (JAX)")

        # Write back to numpy for calibration
        net.W = np.array(state.W)
        scale, frac = calibrate_ee_drive(net, target_frac=0.15)
        mask_ee = net.mask_e_e.astype(bool)
        np.fill_diagonal(mask_ee, False)
        cal_mean = float(net.W_e_e[mask_ee].mean())
        new_w_max = max(cal_mean * 3.0, net.p.w_e_e_max)
        net.p.w_e_e_max = new_w_max
        print(f"    Calibration: scale={scale:.1f}, cal_mean={cal_mean:.4f}, w_max={new_w_max:.4f}")
        state, static = numpy_net_to_jax_state(net)

    # Compute pref AFTER calibration (what matters for F>R)
    rates_post = evaluate_tuning_jax(state, static, thetas_eval, repeats=3)
    osi_post, pref_post = compute_osi(rates_post, thetas_eval)
    print(f"    Mean OSI (post-calibration): {osi_post.mean():.3f}")
    n_pairs = analyze_pref_coverage(pref_post, "Post-calibration prefs")

    if n_pairs == 0:
        print("    SKIP: No neuron pairs for F>R computation")
        return phase_a_mode, None

    # Phase B
    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)

    checkpoints = [0, 100, 200, 400]
    checkpoints = [c for c in checkpoints if c <= n_pres]
    fr_results = []

    W_ee = np.array(state.W_e_e)
    fwd, rev, ratio, _ = compute_fwd_rev_ratio(W_ee, pref_post, SEQ_THETAS)
    fr_results.append((0, ratio, fwd, rev))
    print(f"\n  Phase B ({n_pres} presentations)...")
    print(f"  [pres   0] F>R={ratio:.4f}")

    t0 = time.perf_counter()
    next_cp = 1
    for k in range(1, n_pres + 1):
        state, _ = run_sequence_trial_jax(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
            'ee', ee_A_plus_eff=A_plus, ee_A_minus_eff=A_minus)

        if next_cp < len(checkpoints) and k == checkpoints[next_cp]:
            W_ee = np.array(state.W_e_e)
            fwd, rev, ratio, _ = compute_fwd_rev_ratio(W_ee, pref_post, SEQ_THETAS)
            fr_results.append((k, ratio, fwd, rev))
            print(f"  [pres {k:3d}] F>R={ratio:.4f} (fwd={fwd:.5f}, rev={rev:.5f}) "
                  f"[{time.perf_counter()-t0:.1f}s]")
            next_cp += 1

    trajectory = ' → '.join(f'{r:.3f}' for _, r, _, _ in fr_results)
    print(f"  Trajectory: {trajectory}")
    return phase_a_mode, fr_results


def main():
    print("Comparing JAX vs numpy Phase A: pref distribution and F>R")

    results = {}

    # JAX Phase A (300 seg) — the failing case
    mode, fr = run_test('jax', 300, seed=42, n_pres=400)
    results[f'{mode}_300'] = fr

    # Numpy Phase A (300 seg) — does this fix the distribution?
    mode, fr = run_test('numpy', 300, seed=42, n_pres=400)
    results[f'{mode}_300'] = fr

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, fr in results.items():
        if fr is None:
            print(f"  {label}: SKIP (no pairs)")
        else:
            traj = ' → '.join(f'{r:.3f}' for _, r, _, _ in fr)
            final = fr[-1][1]
            trend = "UP" if final > fr[0][1] + 0.05 else ("DOWN" if final < fr[0][1] - 0.05 else "FLAT")
            print(f"  {label}: {traj} [{trend}]")

    print("\n[DONE]")


if __name__ == '__main__':
    main()

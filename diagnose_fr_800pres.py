#!/usr/bin/env python
"""diagnose_fr_800pres.py — Extended diagnostic: run the FAILING configuration
(frac=0.15, mult=3×, 300 Phase A seg) for the full 800 presentations to see
if F>R reverses at longer training.

Also tests the same configuration with 100 Phase A segments (known to work)
for comparison.
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
import jax.numpy as jnp

GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
THETA_STEP = 180.0 / GOLDEN_RATIO
ELEMENT_MS = 150.0
ITI_MS = 1500.0
SEQ_THETAS = [0.0, 45.0, 90.0, 135.0]
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


def run_condition(phase_a_seg, target_frac, w_max_mult, n_pres, seed=42):
    """Run full pipeline for one condition and return F>R trajectory."""
    label = f"{phase_a_seg}seg_frac{target_frac:.2f}_{w_max_mult:.0f}x"
    print(f"\n{'='*70}")
    print(f"Condition: {label} ({n_pres} presentations)")
    print(f"{'='*70}")

    # Phase A
    p = Params(
        M=16, N=8, seed=seed,
        ee_stdp_enabled=True, ee_connectivity="all_to_all",
        ee_stdp_A_plus=0.005, ee_stdp_A_minus=0.006,
        ee_stdp_weight_dep=True, train_segments=0, segment_ms=300.0,
    )
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)

    print(f"  Phase A ({phase_a_seg} segments)...")
    t0 = time.perf_counter()
    for seg in range(phase_a_seg):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)
    print(f"    Done in {time.perf_counter()-t0:.1f}s")

    # Evaluate tuning
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = evaluate_tuning_jax(state, static, thetas_eval, repeats=2)
    osi_vals, pref = compute_osi(rates, thetas_eval)
    print(f"  Mean OSI: {osi_vals.mean():.3f}")

    # Neuron coverage
    for th in SEQ_THETAS:
        d = np.abs(pref - th)
        d = np.minimum(d, 180 - d)
        n_tuned = np.sum(d < 22.5)
        print(f"    Theta {th:5.1f}°: {n_tuned} neurons")

    # Calibrate
    net.W = np.array(state.W)
    scale, frac = calibrate_ee_drive(net, target_frac=target_frac)
    mask_ee = net.mask_e_e.astype(bool) if hasattr(net, 'mask_e_e') else np.ones((16,16), dtype=bool)
    np.fill_diagonal(mask_ee, False)
    cal_mean = float(net.W_e_e[mask_ee].mean())
    new_w_max = max(cal_mean * w_max_mult, net.p.w_e_e_max)
    net.p.w_e_e_max = new_w_max
    print(f"  Calibration: scale={scale:.1f}, cal_mean={cal_mean:.4f}, w_max={new_w_max:.4f}")

    state, static = numpy_net_to_jax_state(net)
    # Re-compute pref with calibrated state
    rates2 = evaluate_tuning_jax(state, static, thetas_eval, repeats=3)
    _, pref = compute_osi(rates2, thetas_eval)

    # Phase B
    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)

    checkpoints = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800]
    checkpoints = [c for c in checkpoints if c <= n_pres]
    fr_results = []

    W_ee = np.array(state.W_e_e)
    fwd, rev, ratio, n_pairs = compute_fwd_rev_ratio(W_ee, pref, SEQ_THETAS)
    fr_results.append((0, ratio, fwd, rev))
    print(f"\n  Phase B ({n_pres} presentations, {n_pairs} fwd/rev pairs)...")
    print(f"  [pres   0] F>R={ratio:.4f} (fwd={fwd:.5f}, rev={rev:.5f})")

    t0 = time.perf_counter()
    next_cp = 1

    for k in range(1, n_pres + 1):
        state, _ = run_sequence_trial_jax(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
            'ee', ee_A_plus_eff=A_plus, ee_A_minus_eff=A_minus)

        if next_cp < len(checkpoints) and k == checkpoints[next_cp]:
            W_ee = np.array(state.W_e_e)
            off_diag = W_ee[mask_ee]
            fwd, rev, ratio, _ = compute_fwd_rev_ratio(W_ee, pref, SEQ_THETAS)
            fr_results.append((k, ratio, fwd, rev))
            elapsed = time.perf_counter() - t0
            print(f"  [pres {k:3d}] F>R={ratio:.4f} (fwd={fwd:.5f}, rev={rev:.5f}) "
                  f"W: mean={off_diag.mean():.4f} max={off_diag.max():.4f} [{elapsed:.1f}s]")
            next_cp += 1

    print(f"  Total Phase B: {time.perf_counter()-t0:.1f}s")
    return label, fr_results, n_pairs


def main():
    print("Extended F>R diagnostic: 800 presentations, comparing 100 vs 300 Phase A seg")

    results = {}

    # Condition 1: 300 seg, frac=0.15, 3× (the FAILING config from validate_omission_fix.py)
    label, fr, n_pairs = run_condition(300, 0.15, 3.0, 800)
    results[label] = (fr, n_pairs)

    # Condition 2: 100 seg, frac=0.15, 3× (the WORKING config)
    label, fr, n_pairs = run_condition(100, 0.15, 3.0, 800)
    results[label] = (fr, n_pairs)

    # Condition 3: 300 seg, frac=0.15, 2× (testing the original code's multiplier)
    label, fr, n_pairs = run_condition(300, 0.15, 2.0, 800)
    results[label] = (fr, n_pairs)

    # Condition 4: 300 seg, frac=0.10, 3× (lower target)
    label, fr, n_pairs = run_condition(300, 0.10, 3.0, 800)
    results[label] = (fr, n_pairs)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: F>R trajectory across all conditions")
    print("=" * 80)
    for label, (fr, n_pairs) in results.items():
        trajectory = ' → '.join(f'{r:.3f}' for _, r, _, _ in fr)
        final_r = fr[-1][1]
        ok = "PASS (>1.5)" if final_r > 1.5 else ("MARGINAL (>1.0)" if final_r > 1.0 else "FAIL (<1.0)")
        print(f"\n  {label} (n_pairs={n_pairs}):")
        print(f"    {trajectory}")
        print(f"    Final F>R = {final_r:.4f} [{ok}]")

    print("\n[DONE]")
    return 0


if __name__ == '__main__':
    sys.exit(main())

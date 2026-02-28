#!/usr/bin/env python
"""diagnose_fix.py — Test the proposed fix for 300-segment F>R failure.

Root cause: calibrate_ee_drive with target_frac=0.15 pushes E→E weights so
high that recurrent dynamics overwhelm feedforward selectivity, distorting
preferred orientations and creating reverse-favoring STDP dynamics.

Proposed fix:
1. Use pre-calibration preferred orientations for F>R metric (these reflect
   true feedforward selectivity, undistorted by recurrent dynamics)
2. Use lower target_frac (0.03-0.05) to keep E→E weights in a regime where
   feedforward selectivity is preserved
3. Use higher OSI floor (0.5) to prevent calibration from destroying tuning

Tests:
- Various target_frac values with 300 Phase A segments
- Compare pre-cal vs post-cal pref for F>R metric
- Check if lower target_frac produces F>R > 1.0 at 800 presentations
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


def count_pairs(pref, seq_thetas):
    """Count neuron pairs for F>R metric."""
    total = 0
    for ei in range(len(seq_thetas) - 1):
        pre_th, post_th = seq_thetas[ei], seq_thetas[ei + 1]
        d_pre = np.abs(pref - pre_th)
        d_pre = np.minimum(d_pre, 180 - d_pre)
        d_post = np.abs(pref - post_th)
        d_post = np.minimum(d_post, 180 - d_post)
        n_pre = np.sum(d_pre < 22.5)
        n_post = np.sum(d_post < 22.5)
        total += n_pre * n_post
    return total


def run_condition(target_frac, w_max_mult=3.0, osi_floor=0.3, n_pres=800, seed=42):
    """Run a single condition with specified parameters."""
    label = f"frac={target_frac:.2f}_mult={w_max_mult:.0f}x_osi_floor={osi_floor:.1f}"
    print(f"\n{'='*70}")
    print(f"Condition: {label}")
    print(f"{'='*70}")

    # Phase A in JAX (faster, and we'll use pre-cal pref anyway)
    p = Params(
        M=16, N=8, seed=seed,
        ee_stdp_enabled=True, ee_connectivity="all_to_all",
        ee_stdp_A_plus=0.005, ee_stdp_A_minus=0.006,
        ee_stdp_weight_dep=True, train_segments=0, segment_ms=300.0,
    )
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)

    print(f"  Phase A (300 segments, JAX)...")
    for seg in range(300):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)

    # Pre-calibration tuning
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates_pre = evaluate_tuning_jax(state, static, thetas_eval, repeats=3)
    osi_pre, pref_pre = compute_osi(rates_pre, thetas_eval)
    print(f"  Pre-cal OSI: {osi_pre.mean():.3f}, pref coverage: "
          f"{count_pairs(pref_pre, SEQ_THETAS)} pairs")

    # Calibrate
    net.W = np.array(state.W)
    scale, frac = calibrate_ee_drive(net, target_frac=target_frac, osi_floor=osi_floor)
    mask_ee = net.mask_e_e.astype(bool) if hasattr(net, 'mask_e_e') else np.ones((16,16), dtype=bool)
    np.fill_diagonal(mask_ee, False)
    cal_mean = float(net.W_e_e[mask_ee].mean())
    new_w_max = max(cal_mean * w_max_mult, net.p.w_e_e_max)
    net.p.w_e_e_max = new_w_max
    print(f"  Calibration: scale={scale:.1f}, frac={frac:.4f}, "
          f"cal_mean={cal_mean:.4f}, w_max={new_w_max:.4f}")

    # Convert to JAX
    state, static = numpy_net_to_jax_state(net)

    # Post-calibration tuning
    rates_post = evaluate_tuning_jax(state, static, thetas_eval, repeats=3)
    osi_post, pref_post = compute_osi(rates_post, thetas_eval)
    print(f"  Post-cal OSI: {osi_post.mean():.3f}, pref coverage: "
          f"{count_pairs(pref_post, SEQ_THETAS)} pairs")

    # Use pre-calibration pref (true feedforward selectivity)
    pref = pref_pre
    n_pairs = count_pairs(pref, SEQ_THETAS)
    if n_pairs == 0:
        print("  SKIP: 0 pairs with pre-cal pref")
        return label, None, None

    # Phase B
    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)

    checkpoints = [0, 100, 200, 400, 600, 800]
    checkpoints = [c for c in checkpoints if c <= n_pres]
    fr_pre = []   # F>R using pre-cal pref
    fr_post = []  # F>R using post-cal pref

    W_ee = np.array(state.W_e_e)
    fwd_pre, rev_pre, ratio_pre, _ = compute_fwd_rev_ratio(W_ee, pref_pre, SEQ_THETAS)
    fwd_post, rev_post, ratio_post, n_p = compute_fwd_rev_ratio(W_ee, pref_post, SEQ_THETAS)
    fr_pre.append((0, ratio_pre, fwd_pre, rev_pre))
    fr_post.append((0, ratio_post, fwd_post, rev_post))
    print(f"\n  Phase B ({n_pres} presentations)...")
    print(f"  [pres   0] F>R(pre_pref)={ratio_pre:.4f} F>R(post_pref)={ratio_post:.4f}")

    t0 = time.perf_counter()
    next_cp = 1

    for k in range(1, n_pres + 1):
        state, _ = run_sequence_trial_jax(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
            'ee', ee_A_plus_eff=A_plus, ee_A_minus_eff=A_minus)

        if next_cp < len(checkpoints) and k == checkpoints[next_cp]:
            W_ee = np.array(state.W_e_e)
            off_diag = W_ee[mask_ee]
            fwd_pre, rev_pre, ratio_pre, _ = compute_fwd_rev_ratio(W_ee, pref_pre, SEQ_THETAS)
            fwd_post, rev_post, ratio_post, _ = compute_fwd_rev_ratio(W_ee, pref_post, SEQ_THETAS)
            fr_pre.append((k, ratio_pre, fwd_pre, rev_pre))
            fr_post.append((k, ratio_post, fwd_post, rev_post))
            elapsed = time.perf_counter() - t0
            print(f"  [pres {k:3d}] F>R(pre)={ratio_pre:.4f} (fwd={fwd_pre:.5f} rev={rev_pre:.5f})"
                  f"  F>R(post)={ratio_post:.4f}"
                  f"  W: mean={off_diag.mean():.4f} max={off_diag.max():.4f} [{elapsed:.1f}s]")
            next_cp += 1

    return label, fr_pre, fr_post


def main():
    print("Testing proposed fix for 300-segment F>R failure")
    print("Comparing target_frac values with pre-cal vs post-cal pref")

    results = {}

    # Original failing config
    label, fr_pre, fr_post = run_condition(target_frac=0.15, w_max_mult=3.0, osi_floor=0.3)
    results[label] = (fr_pre, fr_post)

    # Fix 1: Much lower target_frac
    label, fr_pre, fr_post = run_condition(target_frac=0.03, w_max_mult=3.0, osi_floor=0.3)
    results[label] = (fr_pre, fr_post)

    # Fix 2: Lower target + tighter OSI floor
    label, fr_pre, fr_post = run_condition(target_frac=0.15, w_max_mult=3.0, osi_floor=0.6)
    results[label] = (fr_pre, fr_post)

    # Fix 3: Very low frac + 2x multiplier
    label, fr_pre, fr_post = run_condition(target_frac=0.03, w_max_mult=2.0, osi_floor=0.3)
    results[label] = (fr_pre, fr_post)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for label, (fr_pre, fr_post) in results.items():
        print(f"\n  {label}:")
        if fr_pre is None:
            print("    SKIP (no pairs)")
            continue
        traj_pre = ' → '.join(f'{r:.3f}' for _, r, _, _ in fr_pre)
        final_pre = fr_pre[-1][1]
        traj_post = ' → '.join(f'{r:.3f}' for _, r, _, _ in fr_post) if fr_post else "N/A"
        final_post = fr_post[-1][1] if fr_post else 0
        status_pre = "PASS(>1.5)" if final_pre > 1.5 else "OK(>1.0)" if final_pre > 1.0 else "FAIL"
        status_post = "PASS(>1.5)" if final_post > 1.5 else "OK(>1.0)" if final_post > 1.0 else "FAIL"
        print(f"    Pre-cal pref : {traj_pre}  [{status_pre}]")
        print(f"    Post-cal pref: {traj_post}  [{status_post}]")

    print("\n[DONE]")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""diagnose_fr_300seg.py — Diagnose why F>R fails with 300 Phase A segments.

Tests multiple calibration targets and w_e_e_max multipliers to find
the parameter regime where F>R > 1.0 after 200 Phase B presentations.

Root cause hypothesis: With 300 segments, calibrated E→E weights are much
higher (scale~1000, mean~2.0) than 100 segments (scale~805, mean~1.61).
The stronger recurrent excitation causes within-element reverberation that
bleeds into the next element window, creating "reverse-causal" STDP timing
that strengthens reverse connections more than forward ones.

Two fixes to test:
1. Lower calibration target_frac (0.05 or 0.10 instead of 0.15)
2. Lower w_e_e_max multiplier (2× instead of 3×)
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

# Constants
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
THETA_STEP = 180.0 / GOLDEN_RATIO
ELEMENT_MS = 150.0
ITI_MS = 1500.0
SEQ_THETAS = [0.0, 45.0, 90.0, 135.0]
CONTRAST = 1.0


def compute_fwd_rev_ratio(W_e_e, pref, seq_thetas):
    """Compute forward/reverse weight asymmetry ratio."""
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
        return 0.0, 0.0, 1.0
    fwd_m = float(np.mean(fwd_ws))
    rev_m = float(np.mean(rev_ws))
    ratio = fwd_m / max(1e-10, rev_m)
    return fwd_m, rev_m, ratio


def run_phase_a_jax(seed=42, M=16, N=8, n_segments=300):
    """Run Phase A in JAX and return (state, static, net) with net having JAX-trained weights."""
    p = Params(
        M=M, N=N, seed=seed,
        ee_stdp_enabled=True,
        ee_connectivity="all_to_all",
        ee_stdp_A_plus=0.005,
        ee_stdp_A_minus=0.006,
        ee_stdp_weight_dep=True,
        train_segments=0,
        segment_ms=300.0,
    )
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)

    for seg in range(n_segments):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)
        if (seg + 1) % 100 == 0:
            print(f"    Phase A: {seg + 1}/{n_segments}")

    # Write back FF weights to net for calibration
    net.W = np.array(state.W)
    return state, static, net


def calibrate_and_convert(net, state_jax, target_frac, w_max_mult):
    """Calibrate E→E drive and convert to JAX state with specified w_max multiplier."""
    print(f"    Calibrating target_frac={target_frac:.2f}, w_max_mult={w_max_mult}x...")

    # Save original W_e_e
    W_e_e_orig = net.W_e_e.copy()

    scale, frac = calibrate_ee_drive(net, target_frac=target_frac)

    # Compute calibrated mean
    if hasattr(net, 'mask_e_e'):
        cal_mean = float(net.W_e_e[net.mask_e_e.astype(bool)].mean())
    else:
        cal_mean = float(net.W_e_e.mean())

    new_w_max = max(cal_mean * w_max_mult, net.p.w_e_e_max)
    net.p.w_e_e_max = new_w_max

    print(f"      scale={scale:.1f}, frac={frac:.4f}, "
          f"cal_mean={cal_mean:.4f}, w_max={new_w_max:.4f}")

    # Convert to JAX (reuse JAX-trained FF weights)
    net.W = np.array(state_jax.W)  # ensure FF weights are from JAX
    state, static = numpy_net_to_jax_state(net)

    # Restore original W_e_e for next iteration
    net.W_e_e = W_e_e_orig
    net.p.w_e_e_max = 0.2  # reset to default

    return state, static, scale, cal_mean, new_w_max


def run_phase_b_diagnostic(state, static, pref, n_presentations=200, label=""):
    """Run Phase B and track F>R at checkpoints."""
    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)

    checkpoints = [0, 50, 100, 150, 200]
    results = []

    # Initial F>R
    W_ee = np.array(state.W_e_e)
    fwd, rev, ratio = compute_fwd_rev_ratio(W_ee, pref, SEQ_THETAS)
    results.append((0, ratio, fwd, rev))

    t0 = time.perf_counter()
    next_cp = 1

    for k in range(1, n_presentations + 1):
        state, _ = run_sequence_trial_jax(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
            'ee', ee_A_plus_eff=A_plus, ee_A_minus_eff=A_minus)

        if next_cp < len(checkpoints) and k == checkpoints[next_cp]:
            W_ee = np.array(state.W_e_e)
            mask = np.array(static.mask_e_e, dtype=bool)
            off_diag = W_ee[mask]
            fwd, rev, ratio = compute_fwd_rev_ratio(W_ee, pref, SEQ_THETAS)
            results.append((k, ratio, fwd, rev))
            elapsed = time.perf_counter() - t0
            print(f"      [{label} pres {k:3d}] F>R={ratio:.4f} "
                  f"(fwd={fwd:.5f}, rev={rev:.5f}) "
                  f"W_ee: mean={off_diag.mean():.5f} max={off_diag.max():.4f} "
                  f"[{elapsed:.1f}s]")
            next_cp += 1

    return results


def main():
    print("=" * 80)
    print("Diagnostic: F>R ratio with 300 Phase A segments")
    print("Testing calibration targets × w_max multipliers")
    print("=" * 80)

    seed = 42
    phase_a_seg = 300

    # ── Phase A (shared across all conditions) ──
    print(f"\n[1] Running Phase A ({phase_a_seg} segments, JAX)...")
    t0 = time.perf_counter()
    state_jax, static_jax, net = run_phase_a_jax(seed=seed, n_segments=phase_a_seg)
    phaseA_time = time.perf_counter() - t0
    print(f"    Phase A done in {phaseA_time:.1f}s")

    # Evaluate tuning
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = evaluate_tuning_jax(state_jax, static_jax, thetas_eval, repeats=2)
    osi_vals, pref = compute_osi(rates, thetas_eval)
    print(f"    Mean OSI: {osi_vals.mean():.3f}")
    print(f"    Preferred orientations: {np.sort(pref)}")

    # Check which sequence elements have neurons tuned near them
    for th in SEQ_THETAS:
        d = np.abs(pref - th)
        d = np.minimum(d, 180 - d)
        n_tuned = np.sum(d < 22.5)
        print(f"    Theta {th:5.1f}°: {n_tuned} neurons within 22.5°")

    # ── Diagnostic: initial E→E weight stats ──
    print(f"\n[2] Initial E→E weight stats (before calibration):")
    W_ee_init = net.W_e_e.copy()
    mask_ee = net.mask_e_e.astype(bool) if hasattr(net, 'mask_e_e') else np.ones_like(W_ee_init, dtype=bool)
    np.fill_diagonal(mask_ee, False)
    off_diag_init = W_ee_init[mask_ee]
    print(f"    mean={off_diag_init.mean():.6f}, std={off_diag_init.std():.6f}, "
          f"max={off_diag_init.max():.6f}")

    # ── Grid search ──
    target_fracs = [0.05, 0.10, 0.15]
    w_max_mults = [2.0, 3.0]

    print(f"\n[3] Grid search: target_frac × w_max_mult")
    print(f"    (each runs 200 Phase B presentations)")

    all_results = {}

    for target_frac in target_fracs:
        for w_max_mult in w_max_mults:
            label = f"frac={target_frac:.2f}_mult={w_max_mult:.0f}x"
            print(f"\n  ── Condition: target_frac={target_frac}, w_max={w_max_mult}× ──")

            # Calibrate (modifies net.W_e_e temporarily)
            state_b, static_b, scale, cal_mean, w_max = calibrate_and_convert(
                net, state_jax, target_frac, w_max_mult)

            # Get pref from the JAX-trained network
            # (pref depends only on FF weights, which are shared)

            # Run Phase B
            results = run_phase_b_diagnostic(
                state_b, static_b, pref, n_presentations=200, label=label)

            all_results[label] = {
                'target_frac': target_frac,
                'w_max_mult': w_max_mult,
                'scale': scale,
                'cal_mean': cal_mean,
                'w_max': w_max,
                'fr_trajectory': results,
            }

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Condition':<30} {'Scale':>6} {'CalMean':>8} {'wMax':>8} "
          f"{'F>R@0':>7} {'F>R@100':>7} {'F>R@200':>7} {'Trend':>8}")
    print("-" * 98)

    for label, res in all_results.items():
        traj = res['fr_trajectory']
        r0 = traj[0][1]
        r100 = next((r for k, r, _, _ in traj if k == 100), None)
        r200 = next((r for k, r, _, _ in traj if k == 200), None)
        trend = "UP" if r200 and r200 > r0 + 0.05 else ("DOWN" if r200 and r200 < r0 - 0.05 else "FLAT")
        ok = "PASS" if r200 and r200 > 1.0 else "FAIL"
        print(f"  {label:<28} {res['scale']:>6.0f} {res['cal_mean']:>8.4f} {res['w_max']:>8.4f} "
              f"{r0:>7.3f} {r100 or 0:>7.3f} {r200 or 0:>7.3f} {trend:>6} [{ok}]")

    # ── Analyze the STDP dynamics for best and worst ──
    print("\n" + "=" * 80)
    print("ANALYSIS: Weight-dependent STDP effective rates at initial weights")
    print("=" * 80)

    for label, res in all_results.items():
        W0 = res['cal_mean']
        wmax = res['w_max']
        A_plus = 0.005
        A_minus = 0.006
        eff_ltp = A_plus * (wmax - W0)
        eff_ltd = A_minus * (W0 - 0.0)
        ratio = eff_ltp / max(1e-10, eff_ltd)
        print(f"  {label:<28}: eff_LTP={eff_ltp:.5f}, eff_LTD={eff_ltd:.5f}, "
              f"LTP/LTD={ratio:.3f}, headroom=(wmax-W0)={wmax-W0:.4f}")

    print("\n[DONE]")
    return 0


if __name__ == '__main__':
    sys.exit(main())

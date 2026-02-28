#!/usr/bin/env python
"""Quick diagnostic: test F>R at different STDP rates.

Fixes scale=700 (best from scale diagnostic) and tests different A_plus/A_minus.
Also tests w_e_e_max multipliers.
"""
import sys
import time
import math
import numpy as np

sys.path.insert(0, '.')

from biologically_plausible_v1_stdp import Params, RgcLgnV1Network, compute_osi
from network_jax import (
    numpy_net_to_jax_state, run_segment_jax, run_sequence_trial_jax,
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
        return 0.0, 0.0, 1.0
    return float(np.mean(fwd_ws)), float(np.mean(rev_ws)), \
           float(np.mean(fwd_ws)) / max(1e-10, float(np.mean(rev_ws)))


def test_config(state_base, static_base, pref, scale, A_plus, A_minus,
                w_max_mult=3.0, n_pres=400):
    """Train and return F>R trajectory."""
    M = int(static_base.M)
    eye_M = jnp.eye(M, dtype=jnp.float32)

    W_e_e_scaled = state_base.W_e_e * scale * (1.0 - eye_M)
    state = state_base._replace(W_e_e=W_e_e_scaled)

    mask_ee = np.array(static_base.mask_e_e).astype(bool)
    cal_mean = float(np.array(W_e_e_scaled)[mask_ee].mean())
    new_w_max = max(cal_mean * w_max_mult, float(static_base.w_e_e_max))
    static = static_base._replace(w_e_e_max=new_w_max)

    checkpoints = [0, 100, 200, 400]
    W_np = np.array(state.W_e_e)
    _, _, ratio0 = compute_fwd_rev_ratio(W_np, pref, SEQ_THETAS)
    results = [(0, ratio0)]
    next_cp = 1

    for k in range(1, n_pres + 1):
        state, _ = run_sequence_trial_jax(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
            'ee', ee_A_plus_eff=A_plus, ee_A_minus_eff=A_minus)
        if next_cp < len(checkpoints) and k == checkpoints[next_cp]:
            W_np = np.array(state.W_e_e)
            _, _, ratio = compute_fwd_rev_ratio(W_np, pref, SEQ_THETAS)
            results.append((k, ratio))
            next_cp += 1

    return results, cal_mean, new_w_max


def main():
    print("=" * 70)
    print("STDP Rate & w_max Diagnostic (400 pres, scale=700)")
    print("=" * 70)

    # Phase A
    print("\nPhase A (100 segments)...")
    p = Params(M=16, N=8, seed=42, ee_stdp_enabled=True,
               ee_connectivity="all_to_all", ee_stdp_A_plus=0.005,
               ee_stdp_A_minus=0.006, ee_stdp_weight_dep=True,
               train_segments=0, segment_ms=300.0)
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)
    for seg in range(100):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)
        if (seg + 1) % 50 == 0:
            print(f"  {seg + 1}/100")

    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = evaluate_tuning_jax(state, static, thetas_eval, repeats=3)
    _, pref = compute_osi(rates, thetas_eval)

    # JIT warmup
    print("JIT warmup...")
    ws = state._replace(W_e_e=state.W_e_e * 500 * (1 - jnp.eye(16, dtype=jnp.float32)))
    ss = static._replace(w_e_e_max=3.0)
    ws, _ = run_sequence_trial_jax(ws, ss, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
                                    'ee', ee_A_plus_eff=0.01, ee_A_minus_eff=0.012)
    jax.block_until_ready(ws.W_e_e)

    SCALE = 700

    # Test 1: Different STDP rate multipliers at scale=700, w_max_mult=3
    print(f"\n--- STDP Rate Sweep (scale={SCALE}, w_max_mult=3.0) ---")
    rate_configs = [
        (0.005, 0.006, "1x"),
        (0.010, 0.012, "2x"),
        (0.015, 0.018, "3x"),
        (0.020, 0.024, "4x"),
    ]
    for A_plus, A_minus, label in rate_configs:
        results, cal_mean, w_max = test_config(
            state, static, pref, SCALE, A_plus, A_minus, w_max_mult=3.0, n_pres=400)
        traj = ' → '.join(f'{r:.3f}' for _, r in results)
        rate_per_100 = (results[-1][1] - results[0][1]) / 4.0
        proj_800 = results[0][1] + rate_per_100 * 8.0
        print(f"  {label:3s} (A+={A_plus:.3f},A-={A_minus:.3f}) | w_max={w_max:.2f} | "
              f"F>R: {traj} | proj800={proj_800:.3f}")

    # Test 2: Different w_max multipliers at scale=700, 2x STDP rates
    print(f"\n--- w_max_mult Sweep (scale={SCALE}, 2x STDP rates) ---")
    wmax_configs = [2.0, 3.0, 4.0, 5.0, 8.0]
    for wm in wmax_configs:
        results, cal_mean, w_max = test_config(
            state, static, pref, SCALE, 0.010, 0.012, w_max_mult=wm, n_pres=400)
        traj = ' → '.join(f'{r:.3f}' for _, r in results)
        rate_per_100 = (results[-1][1] - results[0][1]) / 4.0
        proj_800 = results[0][1] + rate_per_100 * 8.0
        print(f"  w_max_mult={wm:.1f} | w_max={w_max:.2f} | "
              f"F>R: {traj} | proj800={proj_800:.3f}")

    # Test 3: Different scales at the best STDP rate
    print(f"\n--- Scale Sweep (best STDP rate from above) ---")
    best_rate = rate_configs[2]  # try 3x
    A_plus, A_minus = best_rate[0], best_rate[1]
    for scale in [500, 700, 840, 1000]:
        results, cal_mean, w_max = test_config(
            state, static, pref, scale, A_plus, A_minus, w_max_mult=3.0, n_pres=400)
        traj = ' → '.join(f'{r:.3f}' for _, r in results)
        rate_per_100 = (results[-1][1] - results[0][1]) / 4.0
        proj_800 = results[0][1] + rate_per_100 * 8.0
        print(f"  scale={scale:5d} | w_max={w_max:.2f} | "
              f"F>R: {traj} | proj800={proj_800:.3f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

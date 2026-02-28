#!/usr/bin/env python
"""Full 800-presentation F>R diagnostic at key scales.

Uses CORRECT pref (pre-calibration, from feedforward tuning only).
"""
import sys
import time
import math
import numpy as np

sys.path.insert(0, '.')

from biologically_plausible_v1_stdp import Params, RgcLgnV1Network, compute_osi
from network_jax import (
    numpy_net_to_jax_state, run_segment_jax, run_sequence_trial_jax,
    reset_state_jax, evaluate_tuning_jax,
    SimState, StaticConfig,
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
    fwd_m = float(np.mean(fwd_ws))
    rev_m = float(np.mean(rev_ws))
    ratio = fwd_m / max(1e-10, rev_m)
    return fwd_m, rev_m, ratio


def build_network(seed=42, M=16, N=8):
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
    return RgcLgnV1Network(p)


def test_scale_full(scale, state_base, static_base, pref, n_pres=800):
    """Train at given scale for 800 presentations and return F>R trajectory."""
    M = int(static_base.M)
    eye_M = jnp.eye(M, dtype=jnp.float32)

    W_e_e_scaled = state_base.W_e_e * scale * (1.0 - eye_M)
    state = state_base._replace(W_e_e=W_e_e_scaled)

    mask_ee = np.array(static_base.mask_e_e).astype(bool)
    cal_mean = float(np.array(W_e_e_scaled)[mask_ee].mean())
    new_w_max = max(cal_mean * 3.0, float(static_base.w_e_e_max))
    static = static_base._replace(w_e_e_max=new_w_max)

    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)

    checkpoints = [0, 100, 200, 400, 600, 800]
    W_np = np.array(state.W_e_e)
    _, _, ratio0 = compute_fwd_rev_ratio(W_np, pref, SEQ_THETAS)
    results = [(0, ratio0)]

    t0 = time.perf_counter()
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

    elapsed = time.perf_counter() - t0
    return results, elapsed, cal_mean, new_w_max


def main():
    print("=" * 70)
    print("Full 800-presentation F>R Diagnostic (correct pref)")
    print("=" * 70)

    # Phase A
    print("\nPhase A (100 segments)...")
    net = build_network(seed=42)
    state, static = numpy_net_to_jax_state(net)
    for seg in range(100):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)
        if (seg + 1) % 50 == 0:
            print(f"  {seg + 1}/100 done")

    # Evaluate pref orientations (BEFORE any scaling — feedforward tuning only)
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = evaluate_tuning_jax(state, static, thetas_eval, repeats=3)
    osi_vals, pref = compute_osi(rates, thetas_eval)
    print(f"  OSI = {osi_vals.mean():.3f}")
    print(f"  Pref: {pref}")

    # JIT warmup
    print("\nJIT warmup...")
    warmup_state = state._replace(
        W_e_e=state.W_e_e * 500.0 * (1.0 - jnp.eye(16, dtype=jnp.float32)))
    warmup_static = static._replace(w_e_e_max=3.0)
    warmup_state, _ = run_sequence_trial_jax(
        warmup_state, warmup_static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
        'ee', ee_A_plus_eff=0.005, ee_A_minus_eff=0.006)
    jax.block_until_ready(warmup_state.W_e_e)

    # Test scales that gave best growth in 200-pres diagnostic
    test_scales = [500, 700, 840, 1000, 1200]

    print(f"\nTesting {len(test_scales)} scales, 800 presentations each...")
    print("-" * 70)

    for scale in test_scales:
        results, elapsed, cal_mean, w_max = test_scale_full(
            scale, state, static, pref, n_pres=800)
        trajectory = ' → '.join(f'{r:.3f}' for _, r in results)
        print(f"  scale={scale:5d} | cal={cal_mean:.3f} | w_max={w_max:.2f} | "
              f"F>R: {trajectory} | {elapsed:.1f}s")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

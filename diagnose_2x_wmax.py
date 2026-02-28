#!/usr/bin/env python
"""Test F>R with 2x w_e_e_max multiplier (matching numpy code)."""
import sys, time, math
import numpy as np
sys.path.insert(0, '.')
from biologically_plausible_v1_stdp import Params, RgcLgnV1Network, compute_osi
from network_jax import (
    numpy_net_to_jax_state, run_segment_jax, run_sequence_trial_jax,
    evaluate_tuning_jax,
)
import jax, jax.numpy as jnp

THETA_STEP = 180.0 / ((1 + math.sqrt(5)) / 2)
SEQ_THETAS = [0.0, 45.0, 90.0, 135.0]
ELEMENT_MS, ITI_MS, CONTRAST = 150.0, 1500.0, 1.0

def compute_fwd_rev_ratio(W_e_e, pref):
    fwd_ws, rev_ws = [], []
    for ei in range(len(SEQ_THETAS) - 1):
        pre_th, post_th = SEQ_THETAS[ei], SEQ_THETAS[ei + 1]
        d_pre = np.minimum(np.abs(pref - pre_th), 180 - np.abs(pref - pre_th))
        d_post = np.minimum(np.abs(pref - post_th), 180 - np.abs(pref - post_th))
        for pi in np.where(d_post < 22.5)[0]:
            for pj in np.where(d_pre < 22.5)[0]:
                if pi != pj:
                    fwd_ws.append(W_e_e[pi, pj])
                    rev_ws.append(W_e_e[pj, pi])
    if not fwd_ws:
        return 1.0
    return float(np.mean(fwd_ws)) / max(1e-10, float(np.mean(rev_ws)))

def test_config(state_base, static_base, pref, scale, w_mult, n_pres=800):
    M = int(static_base.M)
    eye_M = jnp.eye(M, dtype=jnp.float32)
    W_scaled = state_base.W_e_e * scale * (1.0 - eye_M)
    state = state_base._replace(W_e_e=W_scaled)
    mask_ee = np.array(static_base.mask_e_e).astype(bool)
    cal = float(np.array(W_scaled)[mask_ee].mean())
    w_max = max(cal * w_mult, float(static_base.w_e_e_max))
    static = static_base._replace(w_e_e_max=w_max)
    A_plus, A_minus = float(static.ee_stdp_A_plus), float(static.ee_stdp_A_minus)
    cps = [0, 100, 200, 400, 600, 800]
    results = [(0, compute_fwd_rev_ratio(np.array(state.W_e_e), pref))]
    next_cp = 1
    t0 = time.perf_counter()
    for k in range(1, n_pres + 1):
        state, _ = run_sequence_trial_jax(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
            'ee', ee_A_plus_eff=A_plus, ee_A_minus_eff=A_minus)
        if next_cp < len(cps) and k == cps[next_cp]:
            results.append((k, compute_fwd_rev_ratio(np.array(state.W_e_e), pref)))
            next_cp += 1
    return results, time.perf_counter() - t0, cal, w_max

def main():
    print("2x vs 3x w_e_e_max multiplier F>R comparison")
    print("=" * 70)
    p = Params(M=16, N=8, seed=42, ee_stdp_enabled=True, ee_connectivity="all_to_all",
               ee_stdp_A_plus=0.005, ee_stdp_A_minus=0.006, ee_stdp_weight_dep=True,
               train_segments=0, segment_ms=300.0)
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)
    for seg in range(100):
        state, _ = run_segment_jax(state, static, (seg * THETA_STEP) % 180.0, 1.0, True)
        if (seg + 1) % 50 == 0: print(f"  Phase A: {seg+1}/100")
    rates = evaluate_tuning_jax(state, static, np.linspace(0, 180, 12, endpoint=False), repeats=3)
    _, pref = compute_osi(rates, np.linspace(0, 180, 12, endpoint=False))

    # JIT warmup
    ws = state._replace(W_e_e=state.W_e_e * 500 * (1 - jnp.eye(16, dtype=jnp.float32)))
    ss = static._replace(w_e_e_max=2.0)
    ws, _ = run_sequence_trial_jax(ws, ss, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
                                    'ee', ee_A_plus_eff=0.005, ee_A_minus_eff=0.006)
    jax.block_until_ready(ws.W_e_e)

    configs = [
        (500, 2.0, "scale=500, 2x"),
        (500, 3.0, "scale=500, 3x"),
        (700, 2.0, "scale=700, 2x"),
        (700, 3.0, "scale=700, 3x"),
        (840, 2.0, "scale=840, 2x"),
        (840, 3.0, "scale=840, 3x"),
    ]
    for scale, wm, label in configs:
        results, elapsed, cal, w_max = test_config(state, static, pref, scale, wm, n_pres=800)
        traj = ' → '.join(f'{r:.3f}' for _, r in results)
        print(f"  {label:16s} | cal={cal:.3f} w_max={w_max:.2f} | F>R: {traj} | {elapsed:.0f}s")

if __name__ == "__main__":
    main()

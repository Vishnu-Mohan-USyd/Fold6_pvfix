#!/usr/bin/env python
"""Diagnostic: Test F>R at 2× STDP rates + 2× w_max multiplier.

Hypothesis: With A_plus=0.01, A_minus=0.012 (2× the default sequence rates)
and w_e_e_max = 2× cal_mean, F>R should exceed 1.5 after 800 presentations.

Evidence: 2× rates at scale=700 with 3× w_max gave proj800=1.472.
Since 2× w_max consistently outperforms 3× w_max for F>R, this config
should surpass 1.5.
"""
import sys, time, math
import numpy as np
sys.path.insert(0, '.')
from biologically_plausible_v1_stdp import Params, RgcLgnV1Network, compute_osi
from network_jax import (
    numpy_net_to_jax_state, run_segment_jax, run_sequence_trial_jax,
    evaluate_tuning_jax, calibrate_ee_drive_jax,
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
        return 1.0, 0.0, 0.0
    fwd_m, rev_m = float(np.mean(fwd_ws)), float(np.mean(rev_ws))
    return fwd_m / max(1e-10, rev_m), fwd_m, rev_m

def main():
    print("=" * 70)
    print("Diagnostic: 2x STDP rates + 2x w_max (800 presentations)")
    print("=" * 70)

    # Build network with BOTH 1x and 2x STDP rates stored in Params
    # (only matters for static config — we'll pass rates explicitly)
    p = Params(M=16, N=8, seed=42, ee_stdp_enabled=True, ee_connectivity="all_to_all",
               ee_stdp_A_plus=0.005, ee_stdp_A_minus=0.006, ee_stdp_weight_dep=True,
               train_segments=0, segment_ms=300.0)
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)

    # Phase A
    print("\nPhase A (100 segments)...")
    for seg in range(100):
        state, _ = run_segment_jax(state, static, (seg * THETA_STEP) % 180.0, 1.0, True)
        if (seg + 1) % 50 == 0: print(f"  {seg+1}/100")

    # Pre-calibration pref
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = evaluate_tuning_jax(state, static, thetas_eval, repeats=2)
    osi_vals, pref = compute_osi(rates, thetas_eval)
    print(f"  Mean OSI: {osi_vals.mean():.3f}")

    # Calibrate
    print("Calibrating...")
    scale, frac = calibrate_ee_drive_jax(state, static)
    print(f"  scale={scale:.1f}, frac={frac:.4f}")

    # Apply calibration with 2× w_max
    M = 16
    eye_M = jnp.eye(M, dtype=jnp.float32)
    W_cal = state.W_e_e * scale * (1.0 - eye_M)
    state = state._replace(W_e_e=W_cal)
    mask_ee = np.array(static.mask_e_e).astype(bool)
    cal_mean = float(np.array(W_cal)[mask_ee].mean())
    w_max_2x = max(cal_mean * 2.0, float(static.w_e_e_max))
    static = static._replace(w_e_e_max=w_max_2x)
    print(f"  cal_mean={cal_mean:.4f}, w_max(2x)={w_max_2x:.4f}")

    # JIT warmup
    print("JIT warmup...")
    ws = state._replace(W_e_e=W_cal)
    ws, _ = run_sequence_trial_jax(ws, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
                                    'ee', ee_A_plus_eff=0.01, ee_A_minus_eff=0.012)
    jax.block_until_ready(ws.W_e_e)

    # Test configs: 1× and 2× STDP rates
    configs = [
        (0.005, 0.006, "1x rates"),
        (0.010, 0.012, "2x rates"),
        (0.015, 0.018, "3x rates"),
    ]

    for A_plus, A_minus, label in configs:
        print(f"\n--- {label} (A+={A_plus}, A-={A_minus}) ---")
        s = state._replace(W_e_e=W_cal)  # Reset to calibrated weights
        cps = [0, 100, 200, 400, 600, 800]
        ratio, fwd, rev = compute_fwd_rev_ratio(np.array(s.W_e_e), pref)
        results = [(0, ratio, fwd, rev)]
        next_cp = 1
        t0 = time.perf_counter()

        for k in range(1, 801):
            s, _ = run_sequence_trial_jax(
                s, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
                'ee', ee_A_plus_eff=A_plus, ee_A_minus_eff=A_minus)
            if next_cp < len(cps) and k == cps[next_cp]:
                W_np = np.array(s.W_e_e)
                ratio, fwd, rev = compute_fwd_rev_ratio(W_np, pref)
                off_diag = W_np[mask_ee]
                elapsed = time.perf_counter() - t0
                results.append((k, ratio, fwd, rev))
                print(f"  [pres {k}] F>R={ratio:.4f} (fwd={fwd:.6f}, rev={rev:.6f}), "
                      f"W_ee mean={off_diag.mean():.5f} max={off_diag.max():.4f}, {elapsed:.0f}s")
                next_cp += 1

        traj = ' → '.join(f'{r:.3f}' for _, r, _, _ in results)
        print(f"  Trajectory: {traj}")
        final = results[-1][1]
        print(f"  Final F>R: {final:.4f} {'PASS (>1.5)' if final > 1.5 else 'FAIL (<1.5)'}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

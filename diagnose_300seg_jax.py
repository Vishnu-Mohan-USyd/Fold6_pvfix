#!/usr/bin/env python
"""diagnose_300seg_jax.py — Diagnostic: F>R ratio with 300 Phase A segments (JAX pipeline).

Key question: Does 300 Phase A segments produce F>R > 1.5 at 400 presentations?

Protocol:
  1. Create network (seed=42, M=16, N=8)
  2. Phase A: 300 segments via JAX (run_segment_jax)
  3. Evaluate tuning, get pre-calibration preferred orientations
  4. Calibrate E→E drive (calibrate_ee_drive_jax, target_frac=0.05, osi_floor=0.5)
  5. Apply calibrated scale, w_e_e_max = 3× cal_mean
  6. Phase B: 400 presentations (150ms elements, 1500ms ITI)
  7. Report F>R at checkpoints 0, 50, 100, 200, 400
"""

import sys
import time
import math
import numpy as np

sys.path.insert(0, '.')

from biologically_plausible_v1_stdp import Params, RgcLgnV1Network, compute_osi
from network_jax import (
    numpy_net_to_jax_state,
    run_segment_jax, run_sequence_trial_jax, reset_state_jax,
    evaluate_tuning_jax, calibrate_ee_drive_jax,
    SimState, StaticConfig,
)
import jax
import jax.numpy as jnp

# ── Constants ──────────────────────────────────────────────────────────────

GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
THETA_STEP = 180.0 / GOLDEN_RATIO  # ≈ 111.246°

# Gavornik & Bear (2014) protocol
ELEMENT_MS = 150.0
ITI_MS = 1500.0
SEQ_THETAS = [0.0, 45.0, 90.0, 135.0]
CONTRAST = 1.0

PHASE_A_SEGMENTS = 300
N_PRESENTATIONS = 400
SEED = 42
M = 16
N = 8


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


def main():
    print("=" * 70)
    print(f"Diagnostic: 300 Phase A segments + 400 Phase B presentations (JAX)")
    print(f"seed={SEED}, M={M}, N={N}")
    print("=" * 70)

    t_total = time.perf_counter()

    # ── Step 1: Create network ─────────────────────────────────────────────
    print(f"\n  Step 1: Creating network (seed={SEED})...")
    p = Params(
        M=M, N=N, seed=SEED,
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
    print(f"    Network created, converted to JAX.")

    # ── Step 2: Phase A — 300 segments ─────────────────────────────────────
    print(f"\n  Step 2: Phase A ({PHASE_A_SEGMENTS} segments, JAX)...")
    t0 = time.perf_counter()
    for seg in range(PHASE_A_SEGMENTS):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)
        if (seg + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    Phase A: {seg + 1}/{PHASE_A_SEGMENTS} segments done ({elapsed:.1f}s)")
    phase_a_time = time.perf_counter() - t0
    print(f"    Phase A completed in {phase_a_time:.1f}s")

    # ── Step 3: Evaluate tuning ────────────────────────────────────────────
    print(f"\n  Step 3: Evaluating tuning...")
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = evaluate_tuning_jax(state, static, thetas_eval, repeats=2)
    osi_vals, pref = compute_osi(rates, thetas_eval)
    mean_osi = float(osi_vals.mean())
    print(f"    Mean OSI after Phase A: {mean_osi:.3f}")
    print(f"    Per-neuron OSI: {', '.join(f'{o:.2f}' for o in osi_vals)}")
    print(f"    Preferred orientations: {', '.join(f'{p:.1f}' for p in pref)}")

    # Check how many neurons are well-tuned for each seq element
    for th in SEQ_THETAS:
        d = np.abs(pref - th)
        d = np.minimum(d, 180.0 - d)
        mask = d < 22.5
        n_tuned = int(mask.sum())
        idx = np.where(mask)[0]
        print(f"    Neurons tuned to {th:.0f}°: {n_tuned} {list(idx)}")

    # ── Step 4: Calibrate E→E drive ────────────────────────────────────────
    print(f"\n  Step 4: Calibrating E→E drive (target_frac=0.05, osi_floor=0.5)...")
    scale, frac = calibrate_ee_drive_jax(state, static, target_frac=0.05, osi_floor=0.5)
    print(f"    Calibration: scale={scale:.1f}, frac={frac:.4f}")

    # ── Step 5: Apply calibrated scale, set w_e_e_max = 3× cal_mean ──────
    print(f"\n  Step 5: Applying calibrated scale...")
    eye_M = jnp.eye(M, dtype=jnp.float32)
    W_e_e_calibrated = state.W_e_e * scale * (1.0 - eye_M)
    state = state._replace(W_e_e=W_e_e_calibrated)

    mask_ee = np.array(static.mask_e_e).astype(bool)
    W_e_e_np = np.array(W_e_e_calibrated)
    cal_mean = float(W_e_e_np[mask_ee].mean()) if mask_ee.any() else float(W_e_e_np.mean())
    new_w_max = cal_mean * 3.0
    # Ensure it's at least as large as the current max
    new_w_max = max(new_w_max, float(static.w_e_e_max))
    print(f"    cal_mean={cal_mean:.6f}, w_e_e_max={new_w_max:.6f} (3× cal_mean)")
    print(f"    W_e_e stats: mean={W_e_e_np[mask_ee].mean():.6f}, "
          f"max={W_e_e_np[mask_ee].max():.6f}, min={W_e_e_np[mask_ee].min():.6f}")

    static = static._replace(w_e_e_max=new_w_max)

    # ── Step 6: Phase B — 400 presentations ────────────────────────────────
    print(f"\n  Step 6: Phase B ({N_PRESENTATIONS} presentations, "
          f"element_ms={ELEMENT_MS}, iti_ms={ITI_MS})...")

    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)
    print(f"    A_plus={A_plus}, A_minus={A_minus}")

    checkpoints = [0, 50, 100, 200, 400]
    fr_results = []

    # Initial F>R ratio
    W_ee_np = np.array(state.W_e_e)
    fwd_m, rev_m, ratio = compute_fwd_rev_ratio(W_ee_np, pref, SEQ_THETAS)
    fr_results.append((0, ratio, fwd_m, rev_m))
    print(f"\n    [pres   0] F>R={ratio:.4f} (fwd={fwd_m:.6f}, rev={rev_m:.6f})")

    t0 = time.perf_counter()
    next_cp = 1  # index into checkpoints

    for k in range(1, N_PRESENTATIONS + 1):
        state, _ = run_sequence_trial_jax(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
            'ee',
            ee_A_plus_eff=A_plus,
            ee_A_minus_eff=A_minus,
        )

        if next_cp < len(checkpoints) and k == checkpoints[next_cp]:
            W_ee_np = np.array(state.W_e_e)
            fwd_m, rev_m, ratio = compute_fwd_rev_ratio(W_ee_np, pref, SEQ_THETAS)
            off_diag = W_ee_np[mask_ee]
            fr_results.append((k, ratio, fwd_m, rev_m))
            elapsed = time.perf_counter() - t0
            print(f"    [pres {k:3d}] F>R={ratio:.4f} (fwd={fwd_m:.6f}, rev={rev_m:.6f}), "
                  f"W_ee mean={off_diag.mean():.5f} max={off_diag.max():.4f}, "
                  f"elapsed={elapsed:.1f}s")
            next_cp += 1

    phase_b_time = time.perf_counter() - t0
    total_time = time.perf_counter() - t_total

    # ── Step 7: Report results ─────────────────────────────────────────────
    print(f"\n" + "=" * 70)
    print(f"RESULTS")
    print(f"=" * 70)
    print(f"  Phase A: {PHASE_A_SEGMENTS} segments in {phase_a_time:.1f}s")
    print(f"  Phase B: {N_PRESENTATIONS} presentations in {phase_b_time:.1f}s")
    print(f"  Total: {total_time:.1f}s")
    print(f"  Mean OSI after Phase A: {mean_osi:.3f}")

    print(f"\n  F>R Trajectory:")
    print(f"  {'Pres':>6s}  {'F>R':>8s}  {'Fwd':>10s}  {'Rev':>10s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*10}")
    for pres, ratio, fwd_m, rev_m in fr_results:
        print(f"  {pres:6d}  {ratio:8.4f}  {fwd_m:10.6f}  {rev_m:10.6f}")

    initial_ratio = fr_results[0][1]
    final_ratio = fr_results[-1][1]
    print(f"\n  F>R trajectory: {' → '.join(f'{r:.3f}' for _, r, _, _ in fr_results)}")
    print(f"  Initial F>R: {initial_ratio:.4f}")
    print(f"  Final F>R:   {final_ratio:.4f}")

    if final_ratio > 1.5:
        print(f"\n  ANSWER: YES — F>R = {final_ratio:.4f} > 1.5 at {N_PRESENTATIONS} presentations")
    else:
        print(f"\n  ANSWER: NO — F>R = {final_ratio:.4f} < 1.5 at {N_PRESENTATIONS} presentations")
        print(f"  (Need more presentations or parameter tuning)")

    return 0


if __name__ == '__main__':
    sys.exit(main())

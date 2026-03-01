#!/usr/bin/env python
"""validate_m64_nhc4.py — Multi-HC Phase B at M=64 per HC, n_hc=4.

Same protocol as validate_full_scale.py but with M_PER_HC=64 (M_total=256).
Measures BOTH F>R (weight-based) and omission response (OMR, conductance-based).

Protocol:
  Phase A: 100 golden-angle segments (numpy RNG for orientation diversity)
  Calibration: target_frac=0.05, osi_floor=0.30
  Phase B: 800 sequence presentations with fixed spatial phases
    - F>R measured at checkpoints [100, 200, 400, 600, 800]
    - OMR measured at checkpoints [0, 200, 400, 800]

Pass criteria:
  1. Final F>R median > 1.10
  2. F>R monotonically increasing
  3. Final OMR > 0 (positive omission response)
  4. OMR should increase from checkpoint 0 to 800
"""

import sys
import time
import math
import numpy as np

sys.path.insert(0, '.')

from biologically_plausible_v1_stdp import Params, RgcLgnV1Network, compute_osi
from network_jax import (numpy_net_to_jax_state, run_segment_jax,
                          run_sequence_trial_jax, evaluate_tuning_jax,
                          calibrate_ee_drive_jax, prepare_phaseb_ee,
                          reset_state_jax)
import jax
import jax.numpy as jnp

# ---- Constants ----
GOLDEN = (1 + math.sqrt(5)) / 2
THETA_STEP = 180.0 / GOLDEN

M_PER_HC = 64
N_HC = 4
SEQ_THETAS = [0.0, 45.0, 90.0, 135.0]
THETAS_EVAL = np.linspace(0, 180, 12, endpoint=False)
ELEMENT_MS = 150.0
ITI_MS = 1500.0
A_PLUS = 0.005
A_MINUS = 0.006
TARGET_FRAC = 0.05
N_PRES = 800
FR_CHECKPOINTS = [100, 200, 400, 600, 800]
OMR_CHECKPOINTS = [0, 200, 400, 800]
N_EVAL_TRIALS = 10
SEED = 42
RF_SPACING = 1.0
PHASE_A_SEGMENTS = 100


# ---- F>R helpers (from validate_multihc_tf05.py) ----

def compute_fwd_rev_ratio(W_e_e, pref, seq_thetas):
    """Compute forward/reverse weight ratio for sequential orientation pairs.

    For each sequential pair (theta_i -> theta_{i+1}), identifies neurons
    with preferred orientation within 22.5 deg of each element and measures
    the mean forward vs backward E->E weight.

    Parameters
    ----------
    W_e_e : (M, M) ndarray — E->E weight matrix
    pref : (M,) ndarray — preferred orientations in degrees
    seq_thetas : list of float — sequence orientations

    Returns
    -------
    (fwd_mean, rev_mean, ratio) : tuple of float
    """
    fwd_ws, rev_ws = [], []
    for ei in range(len(seq_thetas) - 1):
        pre_th, post_th = seq_thetas[ei], seq_thetas[ei + 1]
        d_pre = np.abs(pref - pre_th)
        d_pre = np.minimum(d_pre, 180 - d_pre)
        d_post = np.abs(pref - post_th)
        d_post = np.minimum(d_post, 180 - d_post)
        pre_mask = d_pre < 22.5
        post_mask = d_post < 22.5
        for pi in np.where(post_mask)[0]:
            for pj in np.where(pre_mask)[0]:
                if pi != pj:
                    fwd_ws.append(W_e_e[pi, pj])
                    rev_ws.append(W_e_e[pj, pi])
    if len(fwd_ws) == 0:
        return 0.0, 0.0, 1.0
    return (float(np.mean(fwd_ws)),
            float(np.mean(rev_ws)),
            float(np.mean(fwd_ws)) / max(1e-10, float(np.mean(rev_ws))))


def compute_per_hc_fr(W_e_e, pref, n_hc, M_per_hc):
    """Compute F>R ratio for each hypercolumn independently.

    Parameters
    ----------
    W_e_e : (M_total, M_total) ndarray
    pref : (M_total,) ndarray — preferred orientations
    n_hc : int
    M_per_hc : int

    Returns
    -------
    (n_hc,) ndarray of per-HC F>R ratios
    """
    ratios = []
    for hc in range(n_hc):
        s, e = hc * M_per_hc, (hc + 1) * M_per_hc
        _, _, r = compute_fwd_rev_ratio(W_e_e[s:e, s:e], pref[s:e], SEQ_THETAS)
        ratios.append(r)
    return np.array(ratios)


# ---- OMR measurement ----

def evaluate_omr(state, static, phases, n_eval_trials=N_EVAL_TRIALS):
    """Measure omission response using g_exc_ee conductance traces.

    Follows Gavornik & Bear (2014):
      - Trained context: SEQ_THETAS with element 1 (45 deg) omitted
      - Control context: replace pre-omission element (0 deg -> 22.5 deg novel)

    The omission response metric (OMR) is the difference in mean g_exc_ee
    during the omitted element window: trained - control.

    Parameters
    ----------
    state : SimState — network state (weights preserved, dynamics reset each trial)
    static : StaticConfig
    phases : jnp.ndarray — fixed spatial phases, shape (4,)
    n_eval_trials : int — number of evaluation trials to average

    Returns
    -------
    (omr, trained_g_mean, control_g_mean) : tuple of float
        omr = trained_g_mean - control_g_mean
    """
    trained_thetas = list(SEQ_THETAS)             # [0, 45, 90, 135]
    control_thetas = [22.5, 45.0, 90.0, 135.0]   # novel pre-omit element
    omit_index = 1  # omit the 45 deg element

    trained_g_vals = []
    control_g_vals = []

    for trial in range(n_eval_trials):
        # Trained context: familiar sequence with omission
        st_eval = reset_state_jax(state, static)
        _, info_trained = run_sequence_trial_jax(
            st_eval, static, trained_thetas, ELEMENT_MS, ITI_MS, 1.0,
            plastic_mode='none', omit_index=omit_index, phases=phases)
        # info_trained['g_exc_ee_traces']: (n_elem, element_steps, M_total)
        g_traces = np.array(info_trained['g_exc_ee_traces'])
        trained_g_vals.append(float(g_traces[omit_index].mean()))

        # Control context: novel pre-omission element with omission
        st_eval = reset_state_jax(state, static)
        _, info_ctrl = run_sequence_trial_jax(
            st_eval, static, control_thetas, ELEMENT_MS, ITI_MS, 1.0,
            plastic_mode='none', omit_index=omit_index, phases=phases)
        g_traces_ctrl = np.array(info_ctrl['g_exc_ee_traces'])
        control_g_vals.append(float(g_traces_ctrl[omit_index].mean()))

    trained_g_mean = float(np.mean(trained_g_vals))
    control_g_mean = float(np.mean(control_g_vals))
    omr = trained_g_mean - control_g_mean

    return omr, trained_g_mean, control_g_mean


# ---- Main validation ----

def main():
    fixed_phases = jnp.array([0.0, 0.0, 0.0, 0.0])

    print("=" * 80)
    print("FULL-SCALE VALIDATION: F>R + OMR at multi-HC scale")
    print(f"JAX devices: {jax.devices()}")
    print(f"M_per_hc={M_PER_HC}, n_hc={N_HC}, M_total={M_PER_HC * N_HC}")
    print(f"rf_spacing={RF_SPACING}, target_frac={TARGET_FRAC}")
    print(f"N_pres={N_PRES}, seed={SEED}")
    print(f"FR checkpoints: {FR_CHECKPOINTS}")
    print(f"OMR checkpoints: {OMR_CHECKPOINTS}")
    print(f"N_eval_trials (OMR): {N_EVAL_TRIALS}")
    print(f"Phases: FIXED [0, 0, 0, 0]")
    print("=" * 80)

    t_start = time.perf_counter()

    # ---- Build network ----
    p = Params(M=M_PER_HC, N=8, seed=SEED, n_hc=N_HC,
               rf_spacing_pix=RF_SPACING,
               ee_stdp_enabled=True, ee_connectivity="all_to_all",
               ee_stdp_A_plus=A_PLUS, ee_stdp_A_minus=A_MINUS,
               ee_stdp_weight_dep=True, train_segments=0, segment_ms=300.0)
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)

    # ---- Phase A: 100 golden-angle segments ----
    print(f"\n--- Phase A: {PHASE_A_SEGMENTS} golden-angle segments ---")
    t_phaseA = time.perf_counter()
    for seg in range(PHASE_A_SEGMENTS):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)
    print(f"  Phase A complete: {time.perf_counter() - t_phaseA:.1f}s")

    # ---- Pre-calibration tuning ----
    rates = evaluate_tuning_jax(state, static, THETAS_EVAL, repeats=2)
    osi, pref = compute_osi(rates, THETAS_EVAL)
    print(f"  Pre-cal OSI: mean={osi.mean():.3f}")

    # Per-HC coverage report
    for hc in range(N_HC):
        s, e = hc * M_PER_HC, (hc + 1) * M_PER_HC
        cov = {}
        for th in SEQ_THETAS:
            d = np.abs(pref[s:e] - th)
            d = np.minimum(d, 180 - d)
            cov[th] = int(np.sum(d < 22.5))
        min_cov = min(cov.values())
        print(f"    HC{hc}: OSI={osi[s:e].mean():.3f}, coverage={cov}, min={min_cov}")

    # ---- Calibration ----
    print(f"\n--- Calibration: target_frac={TARGET_FRAC} ---")
    scale, _ = calibrate_ee_drive_jax(state, static, target_frac=TARGET_FRAC,
                                       osi_floor=0.30)
    state, static, _, _ = prepare_phaseb_ee(state, static, scale)
    print(f"  scale={scale:.1f}, w_e_e_max={float(static.w_e_e_max):.4f}")

    # Post-calibration OSI
    rates_post = evaluate_tuning_jax(state, static, THETAS_EVAL, repeats=2)
    osi_post, _ = compute_osi(rates_post, THETAS_EVAL)
    print(f"  Post-cal OSI: mean={osi_post.mean():.3f}")
    print(f"  Setup total: {time.perf_counter() - t_start:.1f}s")

    # ---- Phase B: combined F>R + OMR measurement ----
    print(f"\n--- Phase B: {N_PRES} presentations ---")
    print(f"  Measuring F>R at: {FR_CHECKPOINTS}")
    print(f"  Measuring OMR at: {OMR_CHECKPOINTS}")

    fr_results = []   # list of dicts: {pres, fr_median, per_hc, frac_gt1}
    omr_results = []  # list of dicts: {pres, omr, trained_g, control_g}

    # Merge checkpoint sets for the training loop (exclude 0, handled before loop)
    training_checkpoints = sorted(set(
        [c for c in FR_CHECKPOINTS if c > 0] +
        [c for c in OMR_CHECKPOINTS if c > 0]
    ))

    t_phaseB = time.perf_counter()

    # ---- OMR at pres=0 (before any training) ----
    if 0 in OMR_CHECKPOINTS:
        t_omr = time.perf_counter()
        omr_val, trained_g, control_g = evaluate_omr(state, static, fixed_phases)
        omr_results.append({
            'pres': 0, 'omr': omr_val,
            'trained_g': trained_g, 'control_g': control_g
        })
        elapsed_omr = time.perf_counter() - t_omr
        print(f"  [pres    0] OMR={omr_val:+.6f} "
              f"(trained={trained_g:.6f}, ctrl={control_g:.6f}) "
              f"[{elapsed_omr:.1f}s]")

    # ---- Training loop ----
    cp_idx = 0  # index into training_checkpoints

    for k in range(1, N_PRES + 1):
        # Training trial (plastic E->E STDP)
        state, _ = run_sequence_trial_jax(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, 1.0,
            plastic_mode='ee', ee_A_plus_eff=A_PLUS, ee_A_minus_eff=A_MINUS,
            phases=fixed_phases)

        # Check if this is a measurement checkpoint
        if cp_idx < len(training_checkpoints) and k == training_checkpoints[cp_idx]:
            elapsed = time.perf_counter() - t_phaseB
            do_fr = k in FR_CHECKPOINTS
            do_omr = k in OMR_CHECKPOINTS

            parts = []

            # F>R measurement (weight-based, instant)
            if do_fr:
                W = np.array(state.W_e_e)
                per_hc = compute_per_hc_fr(W, pref, N_HC, M_PER_HC)
                fr_med = float(np.median(per_hc))
                frac_gt1 = float(np.mean(per_hc > 1.0))
                hc_str = ' '.join(f'{v:.3f}' for v in per_hc)
                fr_results.append({
                    'pres': k, 'fr_median': fr_med,
                    'per_hc': per_hc.copy(), 'frac_gt1': frac_gt1
                })
                parts.append(
                    f"F>R: med={fr_med:.4f} [{hc_str}] frac>1={frac_gt1:.0%}")

            # OMR measurement (conductance-based, requires eval trials)
            if do_omr:
                t_omr = time.perf_counter()
                omr_val, trained_g, control_g = evaluate_omr(
                    state, static, fixed_phases)
                omr_results.append({
                    'pres': k, 'omr': omr_val,
                    'trained_g': trained_g, 'control_g': control_g
                })
                omr_time = time.perf_counter() - t_omr
                parts.append(
                    f"OMR={omr_val:+.6f} "
                    f"(trained={trained_g:.6f}, ctrl={control_g:.6f}) "
                    f"[omr {omr_time:.1f}s]")

            line = " | ".join(parts)
            print(f"  [pres {k:4d}] {line}  [{elapsed:.1f}s total]")
            cp_idx += 1

    total_phaseB = time.perf_counter() - t_phaseB
    print(f"  Phase B: {total_phaseB:.1f}s "
          f"({total_phaseB / N_PRES * 1000:.0f}ms/pres incl. OMR eval)")

    # ---- Summary ----
    total_time = time.perf_counter() - t_start
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Pre-cal OSI:  {osi.mean():.3f}")
    print(f"  Post-cal OSI: {osi_post.mean():.3f}")

    print(f"\n  F>R trajectory (weight-based, per-HC median):")
    for r in fr_results:
        hc_str = ' '.join(f'HC{i}={v:.3f}' for i, v in enumerate(r['per_hc']))
        print(f"    pres={r['pres']:4d}: med={r['fr_median']:.4f} "
              f"frac>1={r['frac_gt1']:.0%} [{hc_str}]")

    print(f"\n  OMR trajectory (g_exc_ee conductance):")
    for r in omr_results:
        print(f"    pres={r['pres']:4d}: OMR={r['omr']:+.6f} "
              f"(trained={r['trained_g']:.6f}, ctrl={r['control_g']:.6f})")

    print(f"\n  Total time: {total_time:.1f}s")

    # ---- Verdict ----
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")

    # Check 1: Final F>R median > 1.10
    final_fr = fr_results[-1]['fr_median'] if fr_results else 0.0
    pass_fr = final_fr > 1.10
    print(f"  [{'PASS' if pass_fr else 'FAIL'}] Final F>R median = {final_fr:.4f} "
          f"(threshold > 1.10)")

    # Check 2: F>R monotonically increasing
    fr_traj = [r['fr_median'] for r in fr_results]
    monotonic = all(fr_traj[i] <= fr_traj[i + 1]
                    for i in range(len(fr_traj) - 1))
    print(f"  [{'PASS' if monotonic else 'FAIL'}] F>R monotonically increasing: "
          f"{monotonic}")

    # Check 3: Final OMR > 0
    final_omr = omr_results[-1]['omr'] if omr_results else 0.0
    pass_omr = final_omr > 0
    print(f"  [{'PASS' if pass_omr else 'FAIL'}] Final OMR = {final_omr:+.6f} "
          f"(threshold > 0)")

    # Check 4: OMR increases from first to last checkpoint
    if len(omr_results) >= 2:
        omr_increased = omr_results[-1]['omr'] > omr_results[0]['omr']
    else:
        omr_increased = False
    print(f"  [{'PASS' if omr_increased else 'FAIL'}] OMR increased from "
          f"pres {omr_results[0]['pres']} to {omr_results[-1]['pres']}: "
          f"{omr_results[0]['omr']:+.6f} -> {omr_results[-1]['omr']:+.6f}")

    # Per-HC detail at final checkpoint
    if fr_results:
        final = fr_results[-1]
        all_gt1 = all(final['per_hc'] > 1.0)
        print(f"\n  Per-HC at final ({final['pres']} pres):")
        for i, v in enumerate(final['per_hc']):
            print(f"    HC{i}: F>R={v:.4f} {'>' if v > 1.0 else '<='} 1.0")
        print(f"  All HCs > 1.0: {all_gt1}")

    # Overall
    all_pass = pass_fr and monotonic and pass_omr and omr_increased
    print(f"\n  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"\nDone.")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

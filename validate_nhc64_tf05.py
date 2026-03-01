#!/usr/bin/env python
"""validate_nhc64_tf05.py -- Phase B validation at n_hc=64 (8x8 grid) scale.

Tests whether Phase B sequence learning generalizes to a full 64-hypercolumn
retinotopic grid with target_frac=0.05 and fixed phases.

Key parameters:
  - M_PER_HC=36 (6x6 square grid per HC)
  - N_HC=64 (8x8 retinotopic grid), M_total=2304
  - rf_spacing=1.0 (real retinotopy)
  - target_frac=0.05 (preserves post-cal OSI)
  - 800 presentations, checkpoints at [100, 200, 400, 600, 800]
  - Single seed (42) -- n_hc=64 is expensive (~15 min GPU)

Pass criteria:
  1. Per-HC median F>R > 1.05 at final checkpoint
  2. Fraction of HCs with F>R > 1.0 should be > 75%
  3. F>R should be monotonically increasing across checkpoints
"""

import sys, time, math
import numpy as np
sys.path.insert(0, '.')

from biologically_plausible_v1_stdp import Params, RgcLgnV1Network, compute_osi
from network_jax import (numpy_net_to_jax_state, run_segment_jax, run_sequence_trial_jax,
                          evaluate_tuning_jax, calibrate_ee_drive_jax, prepare_phaseb_ee,
                          get_flat_W_e_e_numpy)
import jax
import jax.numpy as jnp

# ── Constants ──────────────────────────────────────────────────────────────
GOLDEN = (1 + math.sqrt(5)) / 2
THETA_STEP = 180.0 / GOLDEN
M_PER_HC = 36
N_HC = 64
SEQ_THETAS = [0.0, 45.0, 90.0, 135.0]
THETAS_EVAL = np.linspace(0, 180, 12, endpoint=False)
ELEMENT_MS, ITI_MS = 150.0, 1500.0
A_PLUS, A_MINUS = 0.005, 0.006
TARGET_FRAC = 0.05
N_PRES = 800
CHECKPOINTS = [100, 200, 400, 600, 800]
SEED = 42
PHASE_A_SEGMENTS = 100


# ── Helpers ────────────────────────────────────────────────────────────────
def compute_fwd_rev_ratio(W_e_e, pref, seq_thetas):
    """Compute forward/reverse weight ratio for a single HC's E-E weight block.

    Parameters
    ----------
    W_e_e : ndarray, shape (M_per_hc, M_per_hc)
        Intra-HC excitatory-to-excitatory weight matrix.
    pref : ndarray, shape (M_per_hc,)
        Preferred orientation (degrees) for each neuron.
    seq_thetas : list of float
        Sequence element orientations.

    Returns
    -------
    fwd_mean : float
        Mean forward weight (pre->post in sequence order).
    rev_mean : float
        Mean reverse weight (post->pre in sequence order).
    ratio : float
        fwd_mean / rev_mean.
    """
    fwd_ws, rev_ws = [], []
    for ei in range(len(seq_thetas) - 1):
        pre_th, post_th = seq_thetas[ei], seq_thetas[ei + 1]
        d_pre = np.abs(pref - pre_th); d_pre = np.minimum(d_pre, 180 - d_pre)
        d_post = np.abs(pref - post_th); d_post = np.minimum(d_post, 180 - d_post)
        pre_mask = d_pre < 22.5; post_mask = d_post < 22.5
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
    W_e_e : ndarray, shape (M_total, M_total)
        Full excitatory weight matrix.
    pref : ndarray, shape (M_total,)
        Preferred orientations for all neurons.
    n_hc : int
        Number of hypercolumns.
    M_per_hc : int
        Neurons per hypercolumn.

    Returns
    -------
    ratios : ndarray, shape (n_hc,)
        F>R ratio for each HC.
    """
    ratios = []
    for hc in range(n_hc):
        s, e = hc * M_per_hc, (hc + 1) * M_per_hc
        _, _, r = compute_fwd_rev_ratio(W_e_e[s:e, s:e], pref[s:e], SEQ_THETAS)
        ratios.append(r)
    return np.array(ratios)


def orientation_coverage_summary(pref, n_hc, M_per_hc, seq_thetas, threshold_deg=22.5):
    """Report how many HCs have at least 1 neuron within threshold of each seq theta.

    Parameters
    ----------
    pref : ndarray, shape (M_total,)
        Preferred orientations.
    n_hc : int
        Number of hypercolumns.
    M_per_hc : int
        Neurons per HC.
    seq_thetas : list of float
        Sequence orientations to check coverage for.
    threshold_deg : float
        Circular distance threshold in degrees.

    Returns
    -------
    coverage_matrix : ndarray, shape (n_hc, len(seq_thetas))
        Number of neurons in each HC within threshold of each seq theta.
    fully_covered_count : int
        Number of HCs with >= 1 neuron for ALL seq thetas.
    """
    n_thetas = len(seq_thetas)
    coverage = np.zeros((n_hc, n_thetas), dtype=int)
    for hc in range(n_hc):
        s, e = hc * M_per_hc, (hc + 1) * M_per_hc
        hc_pref = pref[s:e]
        for ti, th in enumerate(seq_thetas):
            d = np.abs(hc_pref - th)
            d = np.minimum(d, 180 - d)
            coverage[hc, ti] = int(np.sum(d < threshold_deg))
    # A HC is "fully covered" if it has >= 1 neuron for every seq theta
    fully_covered = np.all(coverage >= 1, axis=1)
    return coverage, int(np.sum(fully_covered))


def fr_distribution_string(per_hc_fr):
    """Format F>R distribution stats as a compact string.

    Parameters
    ----------
    per_hc_fr : ndarray, shape (n_hc,)

    Returns
    -------
    str
        Formatted summary with min, Q1, median, mean, Q3, max, frac>1.
    """
    q1, med, q3 = np.percentile(per_hc_fr, [25, 50, 75])
    return (f"min={per_hc_fr.min():.4f} Q1={q1:.4f} med={med:.4f} "
            f"mean={per_hc_fr.mean():.4f} Q3={q3:.4f} max={per_hc_fr.max():.4f} "
            f"frac>1={np.mean(per_hc_fr > 1.0):.1%}")


def main():
    print("=" * 80)
    print("PHASE B VALIDATION: n_hc=64 (8x8 grid), target_frac=0.05")
    print(f"JAX devices: {jax.devices()}")
    print(f"M_per_hc={M_PER_HC}, N_HC={N_HC}, M_total={M_PER_HC * N_HC}")
    print(f"rf_spacing=1.0, target_frac={TARGET_FRAC}, seed={SEED}")
    print(f"Phase A: {PHASE_A_SEGMENTS} golden-angle segments")
    print(f"Phase B: {N_PRES} presentations, checkpoints={CHECKPOINTS}")
    print(f"SEQ_THETAS={SEQ_THETAS}, ELEMENT_MS={ELEMENT_MS}, ITI_MS={ITI_MS}")
    print(f"A_PLUS={A_PLUS}, A_MINUS={A_MINUS}")
    print("=" * 80)

    t_start = time.perf_counter()
    fixed_phases = jnp.array([0.0, 0.0, 0.0, 0.0])

    # ── Build network ──────────────────────────────────────────────────────
    print("\n--- Building network ---")
    p = Params(M=M_PER_HC, N=8, seed=SEED, n_hc=N_HC,
               rf_spacing_pix=1.0,
               ee_stdp_enabled=True, ee_connectivity="all_to_all",
               ee_stdp_A_plus=A_PLUS, ee_stdp_A_minus=A_MINUS,
               ee_stdp_weight_dep=True, train_segments=0, segment_ms=300.0)
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)
    print(f"  Network built: {time.perf_counter() - t_start:.1f}s")

    # ── Phase A: golden-angle training ─────────────────────────────────────
    print(f"\n--- Phase A: {PHASE_A_SEGMENTS} golden-angle segments ---")
    t_phase_a = time.perf_counter()
    for seg in range(PHASE_A_SEGMENTS):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)
        if (seg + 1) % 25 == 0:
            print(f"  seg {seg + 1}/{PHASE_A_SEGMENTS} ({time.perf_counter() - t_phase_a:.1f}s)")
    phase_a_time = time.perf_counter() - t_phase_a
    print(f"  Phase A complete: {phase_a_time:.1f}s ({phase_a_time / PHASE_A_SEGMENTS * 1000:.0f}ms/seg)")

    # ── Evaluate tuning ───────────────────────────────────────────────────
    print("\n--- Tuning evaluation ---")
    rates = evaluate_tuning_jax(state, static, THETAS_EVAL, repeats=2)
    osi, pref = compute_osi(rates, THETAS_EVAL)
    M_total = M_PER_HC * N_HC

    # Per-HC OSI summary
    hc_osi_means = np.array([osi[hc * M_PER_HC:(hc + 1) * M_PER_HC].mean() for hc in range(N_HC)])
    print(f"  Pre-cal OSI: global_mean={osi.mean():.3f}")
    print(f"  Per-HC OSI: min={hc_osi_means.min():.3f} mean={hc_osi_means.mean():.3f} "
          f"max={hc_osi_means.max():.3f}")
    # Show a few representative HCs (corners + center of 8x8 grid)
    representative_hcs = [0, 7, 28, 35, 56, 63]  # corners + center-ish
    for hc in representative_hcs:
        s, e = hc * M_PER_HC, (hc + 1) * M_PER_HC
        print(f"    HC{hc:2d}: OSI={osi[s:e].mean():.3f}")

    # ── Orientation coverage ──────────────────────────────────────────────
    print("\n--- Orientation coverage ---")
    coverage, n_fully_covered = orientation_coverage_summary(pref, N_HC, M_PER_HC, SEQ_THETAS)
    print(f"  HCs with full coverage (>= 1 neuron per seq theta): {n_fully_covered}/{N_HC}")
    # Per-theta summary across all HCs
    for ti, th in enumerate(SEQ_THETAS):
        col = coverage[:, ti]
        n_covered = int(np.sum(col >= 1))
        print(f"    theta={th:5.1f}: {n_covered}/{N_HC} HCs covered, "
              f"neurons/HC: min={col.min()} mean={col.mean():.1f} max={col.max()}")

    # ── Calibrate E->E drive ──────────────────────────────────────────────
    print(f"\n--- Calibration (target_frac={TARGET_FRAC}) ---")
    scale, _ = calibrate_ee_drive_jax(state, static, target_frac=TARGET_FRAC, osi_floor=0.30)
    state, static, _, _ = prepare_phaseb_ee(state, static, scale)
    print(f"  scale={scale:.1f}, w_e_e_max={float(static.w_e_e_max):.6f}")

    # Post-cal OSI
    rates_post = evaluate_tuning_jax(state, static, THETAS_EVAL, repeats=2)
    osi_post, pref_post = compute_osi(rates_post, THETAS_EVAL)
    hc_osi_post = np.array([osi_post[hc * M_PER_HC:(hc + 1) * M_PER_HC].mean() for hc in range(N_HC)])
    print(f"  Post-cal OSI: global_mean={osi_post.mean():.3f}")
    print(f"  Per-HC OSI: min={hc_osi_post.min():.3f} mean={hc_osi_post.mean():.3f} "
          f"max={hc_osi_post.max():.3f}")

    setup_time = time.perf_counter() - t_start
    print(f"\n  Total setup time: {setup_time:.1f}s")

    # ── Phase B: sequence learning ────────────────────────────────────────
    print(f"\n--- Phase B: {N_PRES} presentations ---")
    checkpoint_results = []
    cp_idx = 0
    t_phase_b = time.perf_counter()

    for k in range(1, N_PRES + 1):
        state, _ = run_sequence_trial_jax(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, 1.0,
            plastic_mode='ee', ee_A_plus_eff=A_PLUS, ee_A_minus_eff=A_MINUS,
            phases=fixed_phases)

        if cp_idx < len(CHECKPOINTS) and k == CHECKPOINTS[cp_idx]:
            W = get_flat_W_e_e_numpy(state, static)
            elapsed = time.perf_counter() - t_phase_b

            per_hc_fr = compute_per_hc_fr(W, pref, N_HC, M_PER_HC)
            dist_str = fr_distribution_string(per_hc_fr)
            print(f"  [pres {k:4d}] {dist_str}  ({elapsed:.1f}s)")

            checkpoint_results.append({
                'pres': k,
                'per_hc_fr': per_hc_fr.copy(),
                'fr_median': float(np.median(per_hc_fr)),
                'fr_mean': float(np.mean(per_hc_fr)),
                'frac_gt1': float(np.mean(per_hc_fr > 1.0)),
            })
            cp_idx += 1

        # Progress every 100 presentations (non-checkpoint)
        elif k % 100 == 0:
            elapsed = time.perf_counter() - t_phase_b
            print(f"  [pres {k:4d}] ... ({elapsed:.1f}s)")

    phase_b_time = time.perf_counter() - t_phase_b
    print(f"  Phase B complete: {phase_b_time:.1f}s ({phase_b_time / N_PRES * 1000:.0f}ms/pres)")

    total_time = time.perf_counter() - t_start
    print(f"  Total wall time: {total_time:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Pre-cal OSI:  global={osi.mean():.3f}, per-HC: [{hc_osi_means.min():.3f} - {hc_osi_means.max():.3f}]")
    print(f"  Post-cal OSI: global={osi_post.mean():.3f}, per-HC: [{hc_osi_post.min():.3f} - {hc_osi_post.max():.3f}]")
    print(f"  Orientation coverage: {n_fully_covered}/{N_HC} HCs fully covered")
    print()
    print("  F>R trajectory:")
    for r in checkpoint_results:
        print(f"    pres={r['pres']:4d}: med={r['fr_median']:.4f} mean={r['fr_mean']:.4f} "
              f"frac>1={r['frac_gt1']:.1%}")

    if checkpoint_results:
        final = checkpoint_results[-1]
        print(f"\n  Final ({final['pres']} pres):")
        print(f"    Median F>R: {final['fr_median']:.4f}")
        print(f"    Mean F>R:   {final['fr_mean']:.4f}")
        print(f"    Frac > 1.0: {final['frac_gt1']:.1%}")
        print(f"    Per-HC F>R distribution:")
        per_hc = final['per_hc_fr']
        print(f"      {fr_distribution_string(per_hc)}")
        # Show representative HCs
        for hc in representative_hcs:
            print(f"      HC{hc:2d}: F>R={per_hc[hc]:.4f}")

    # ── Verdict ───────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")

    if not checkpoint_results:
        print("  NO RESULTS -- something went wrong")
        sys.exit(1)

    final = checkpoint_results[-1]
    fr_median = final['fr_median']
    frac_gt1 = final['frac_gt1']

    # Check monotonicity of median F>R across checkpoints
    medians = [r['fr_median'] for r in checkpoint_results]
    monotonic = all(medians[i] <= medians[i + 1] for i in range(len(medians) - 1))

    test1_pass = fr_median > 1.05
    test2_pass = frac_gt1 > 0.75
    test3_pass = monotonic

    print(f"  Test 1 (median F>R > 1.05):    {fr_median:.4f}  {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Test 2 (frac HCs F>R > 1.0 > 75%): {frac_gt1:.1%}  {'PASS' if test2_pass else 'FAIL'}")
    print(f"  Test 3 (monotonic F>R):        {medians}  {'PASS' if test3_pass else 'FAIL'}")

    all_pass = test1_pass and test2_pass and test3_pass
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")

    if not all_pass:
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()

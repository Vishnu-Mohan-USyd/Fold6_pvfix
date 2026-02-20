#!/usr/bin/env python3
"""Validate that additive STDP produces strong forward > reverse in sequence experiment.

This script directly tests the core STDP mechanism without the full experiment overhead,
then runs a medium-length experiment to confirm functional results.
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from biologically_plausible_v1_stdp import (
    Params, RgcLgnV1Network, compute_osi,
    run_grating_element, run_blank_element, run_sequence_trial,
    calibrate_ee_drive, compute_sequence_metrics,
    _run_eval_conditions,
)

def main():
    print("=" * 70)
    print("VALIDATION: Additive STDP sequence learning")
    print("=" * 70)

    # --- Setup ---
    M = 32
    N = 8
    seed = 1
    seq_thetas = [0.0, 45.0, 90.0, 135.0]
    element_ms = 30.0
    iti_ms = 200.0
    contrast = 2.0
    phase_a_segs = 250
    n_presentations = 600  # enough for convergence

    p = Params(
        N=N, M=M, seed=seed,
        train_segments=0,
        train_stimulus="grating",
        train_contrast=contrast,
        ee_stdp_enabled=True,
        ee_connectivity="all_to_all",
        ee_stdp_A_plus=0.002,
        ee_stdp_A_minus=0.0024,
        ee_stdp_weight_dep=False,  # ADDITIVE STDP
    )
    net = RgcLgnV1Network(p, init_mode="random")
    rng = np.random.default_rng(seed + 31337)

    # --- Phase A: develop receptive fields ---
    print(f"\n[Phase A] Training feedforward STDP ({phase_a_segs} segments)...")
    net.ff_plastic_enabled = True
    net.ee_stdp_active = False
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    theta_step = 180.0 / phi
    theta_offset = float(net.rng.uniform(0.0, 180.0))
    for s in range(1, phase_a_segs + 1):
        th = float((theta_offset + (s - 1) * theta_step) % 180.0)
        net.run_segment(th, plastic=True, contrast=contrast)
        if s % 50 == 0 or s == phase_a_segs:
            thetas_eval = np.linspace(0, 180, 12, endpoint=False)
            rates = net.evaluate_tuning(thetas_eval, repeats=3)
            osi, pref = compute_osi(rates, thetas_eval)
            print(f"  seg {s}/{phase_a_segs}: OSI={osi.mean():.3f}, rate={rates.mean():.2f} Hz")

    # --- Calibrate E→E drive ---
    print(f"\n[Calibrate] Auto-calibrating E→E drive to 0.15...")
    scale, frac = calibrate_ee_drive(net, 0.15)
    off_diag = net.W_e_e[net.mask_e_e]
    cal_mean = float(off_diag.mean())
    print(f"  scale={scale:.1f}, frac={frac:.4f}, W_e_e mean={cal_mean:.5f}")
    # Keep w_e_e_max at default 0.2 (for additive STDP, hard bounds are the constraint)
    print(f"  w_e_e_max = {p.w_e_e_max}")

    # --- Assign neurons to groups ---
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = net.evaluate_tuning(thetas_eval, repeats=5)
    osi_pre, pref_pre = compute_osi(rates, thetas_eval)
    print(f"\n[Pre-training] OSI={osi_pre.mean():.3f}, rate={rates.mean():.2f} Hz")

    # Determine which neurons prefer which sequence orientation
    groups = {}
    for i, th in enumerate(seq_thetas):
        # Find neurons whose preferred orientation is closest to th
        diffs = np.abs(pref_pre - th)
        diffs = np.minimum(diffs, 180 - diffs)  # circular distance
        mask = diffs < 22.5  # within ±22.5° of the target
        groups[th] = np.where(mask)[0]
        print(f"  Group {th:5.1f}°: {len(groups[th])} neurons (indices: {groups[th][:5]}...)")

    # Record initial W_e_e for forward/reverse connections
    W_init = net.W_e_e.copy()

    # --- Phase B: sequence training ---
    print(f"\n[Phase B] Sequence training ({n_presentations} presentations)...")
    net.ff_plastic_enabled = False
    net.ee_stdp_active = True
    net._ee_stdp_ramp_factor = 1.0

    log_interval = max(1, n_presentations // 10)
    for k in range(1, n_presentations + 1):
        run_sequence_trial(
            net, seq_thetas, element_ms, iti_ms,
            contrast, plastic=True, record=False,
            vep_mode="spikes",
        )

        if k % log_interval == 0:
            drive_frac, _ = net.get_drive_fraction()
            net.reset_drive_accumulators()
            w_mean = float(net.W_e_e[net.mask_e_e].mean())
            w_max_val = float(net.W_e_e.max())

            # Compute forward vs reverse weight means
            fwd_weights = []
            rev_weights = []
            for i in range(len(seq_thetas) - 1):
                pre_group = groups[seq_thetas[i]]
                post_group = groups[seq_thetas[i + 1]]
                if len(pre_group) > 0 and len(post_group) > 0:
                    for post_idx in post_group:
                        for pre_idx in pre_group:
                            if post_idx != pre_idx:
                                fwd_weights.append(net.W_e_e[post_idx, pre_idx])
                                rev_weights.append(net.W_e_e[pre_idx, post_idx])

            fwd_mean = np.mean(fwd_weights) if fwd_weights else 0.0
            rev_mean = np.mean(rev_weights) if rev_weights else 0.0
            ratio = fwd_mean / max(1e-10, rev_mean)

            print(f"  pres {k:4d}/{n_presentations}: drive={drive_frac:.4f}, "
                  f"W_mean={w_mean:.5f}, W_max={w_max_val:.4f}, "
                  f"fwd={fwd_mean:.5f}, rev={rev_mean:.5f}, ratio={ratio:.3f}")

    # --- Final weight analysis ---
    print(f"\n[Analysis] Final weight matrix statistics:")
    W_final = net.W_e_e.copy()
    dW = W_final - W_init
    print(f"  Total |ΔW| > 1e-4: {(np.abs(dW) > 1e-4).sum()} / {M*M}")
    print(f"  W_e_e mean: {W_init[net.mask_e_e].mean():.5f} → {W_final[net.mask_e_e].mean():.5f}")
    print(f"  W_e_e max: {W_init.max():.5f} → {W_final.max():.5f}")

    # Forward vs reverse for each adjacent pair
    print(f"\n  Per-transition weight analysis:")
    all_fwd = []
    all_rev = []
    for i in range(len(seq_thetas) - 1):
        pre_group = groups[seq_thetas[i]]
        post_group = groups[seq_thetas[i + 1]]
        if len(pre_group) > 0 and len(post_group) > 0:
            fwd = [W_final[post, pre] for post in post_group for pre in pre_group if post != pre]
            rev = [W_final[pre, post] for post in post_group for pre in pre_group if post != pre]
            all_fwd.extend(fwd)
            all_rev.extend(rev)
            fwd_m = np.mean(fwd) if fwd else 0.0
            rev_m = np.mean(rev) if rev else 0.0
            print(f"    {seq_thetas[i]:5.1f}°→{seq_thetas[i+1]:5.1f}°: "
                  f"fwd={fwd_m:.5f}, rev={rev_m:.5f}, "
                  f"ratio={fwd_m / max(1e-10, rev_m):.3f}")

    fwd_mean = np.mean(all_fwd) if all_fwd else 0.0
    rev_mean = np.mean(all_rev) if all_rev else 0.0
    print(f"\n  Overall forward mean: {fwd_mean:.5f}")
    print(f"  Overall reverse mean: {rev_mean:.5f}")
    print(f"  Forward/Reverse ratio: {fwd_mean / max(1e-10, rev_mean):.3f}")

    # --- Evaluate VEP responses ---
    print(f"\n[Evaluation] Testing trained vs reverse VEP...")
    # Trained order
    trained_counts = []
    reverse_counts = []
    test_repeats = 30
    for rep in range(test_repeats):
        # Trained order
        r = run_sequence_trial(net, seq_thetas, element_ms, iti_ms,
                               contrast, plastic=False, record=True, vep_mode="i_exc")
        trained_counts.append(sum(c.sum() for c in r["element_counts"]))
        # Reverse order
        r_rev = run_sequence_trial(net, seq_thetas[::-1], element_ms, iti_ms,
                                    contrast, plastic=False, record=True, vep_mode="i_exc")
        reverse_counts.append(sum(c.sum() for c in r_rev["element_counts"]))

    trained_mean = np.mean(trained_counts)
    reverse_mean = np.mean(reverse_counts)
    trained_std = np.std(trained_counts) / np.sqrt(test_repeats)
    reverse_std = np.std(reverse_counts) / np.sqrt(test_repeats)

    # Simple t-test
    pooled_se = np.sqrt(np.var(trained_counts) / test_repeats + np.var(reverse_counts) / test_repeats)
    t_stat = (trained_mean - reverse_mean) / max(1e-10, pooled_se)

    print(f"  Trained spikes: {trained_mean:.1f} ± {trained_std:.1f}")
    print(f"  Reverse spikes: {reverse_mean:.1f} ± {reverse_std:.1f}")
    print(f"  Ratio (trained/reverse): {trained_mean / max(1e-10, reverse_mean):.3f}")
    print(f"  t-statistic: {t_stat:.3f}")

    # --- Check omission ---
    print(f"\n[Omission] Testing omission prediction...")
    omit_traces = []
    no_omit_traces = []
    for rep in range(test_repeats):
        # With omission of element 2 (90°)
        r_omit = run_sequence_trial(net, seq_thetas, element_ms, iti_ms,
                                     contrast, plastic=False, record=True,
                                     omit_index=2, vep_mode="i_exc")
        omit_traces.append(r_omit["element_traces"][2])  # trace during omitted element

        # Full sequence (no omission) - element 2 trace for comparison
        r_full = run_sequence_trial(net, seq_thetas, element_ms, iti_ms,
                                     contrast, plastic=False, record=True, vep_mode="i_exc")
        no_omit_traces.append(r_full["element_traces"][2])

    omit_mean = np.mean([np.mean(t) for t in omit_traces])
    full_mean = np.mean([np.mean(t) for t in no_omit_traces])
    print(f"  Omitted element mean i_exc: {omit_mean:.4f}")
    print(f"  Full element mean i_exc: {full_mean:.4f}")
    print(f"  Prediction signal (omit/full): {omit_mean / max(1e-10, full_mean):.3f}")

    # --- OSI preservation ---
    rates_post = net.evaluate_tuning(thetas_eval, repeats=5)
    osi_post, _ = compute_osi(rates_post, thetas_eval)
    print(f"\n[OSI] Post-training: {osi_post.mean():.3f} (pre: {osi_pre.mean():.3f})")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    fwd_rev_pass = fwd_mean > rev_mean * 1.5
    spike_pass = trained_mean > reverse_mean
    osi_pass = osi_post.mean() > 0.3
    omit_pass = omit_mean > 0.0

    print(f"  Forward > Reverse (weights, >1.5x): {'PASS' if fwd_rev_pass else 'FAIL'} "
          f"(ratio={fwd_mean / max(1e-10, rev_mean):.3f})")
    print(f"  Trained > Reverse (spikes):         {'PASS' if spike_pass else 'FAIL'} "
          f"(ratio={trained_mean / max(1e-10, reverse_mean):.3f})")
    print(f"  OSI preserved (>0.3):               {'PASS' if osi_pass else 'FAIL'} "
          f"(OSI={osi_post.mean():.3f})")
    print(f"  Omission prediction signal:         {'PASS' if omit_pass else 'FAIL'} "
          f"(signal={omit_mean:.4f})")
    print(f"{'=' * 70}")

    return 0 if (fwd_rev_pass and spike_pass and osi_pass) else 1


if __name__ == "__main__":
    sys.exit(main())

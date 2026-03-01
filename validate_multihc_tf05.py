#!/usr/bin/env python
"""validate_multihc_tf05.py — Multi-HC Phase B with target_frac=0.05.

Key discovery: target_frac=0.15 (default) causes post-cal OSI collapse at M=36
(OSI drops from 0.83→0.42), degrading temporal separation during Phase B.
target_frac=0.05 preserves post-cal OSI≈0.75, giving F>R=1.263 at M=36 n_hc=1.

This test applies the fix at n_hc=4, rf=1.0 with fixed phases.

Conditions:
  A: M=36, n_hc=4, rf=1.0, target_frac=0.05, FIXED phases (key experiment)
  B: M=36, n_hc=1, rf=0, target_frac=0.05, random phases (baseline)

Two seeds: 42, 137.
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

GOLDEN = (1 + math.sqrt(5)) / 2
THETA_STEP = 180.0 / GOLDEN
M_PER_HC = 36
N_HC = 4
SEQ_THETAS = [0.0, 45.0, 90.0, 135.0]
THETAS_EVAL = np.linspace(0, 180, 12, endpoint=False)
ELEMENT_MS, ITI_MS = 150.0, 1500.0
A_PLUS, A_MINUS = 0.005, 0.006
TARGET_FRAC = 0.05
N_PRES = 800
CHECKPOINTS = [100, 200, 400, 600, 800]
SEEDS = [42, 137]


def compute_fwd_rev_ratio(W_e_e, pref, seq_thetas):
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
    return float(np.mean(fwd_ws)), float(np.mean(rev_ws)), float(np.mean(fwd_ws)) / max(1e-10, float(np.mean(rev_ws)))


def compute_per_hc_fr(W_e_e, pref, n_hc, M_per_hc):
    ratios = []
    for hc in range(n_hc):
        s, e = hc * M_per_hc, (hc + 1) * M_per_hc
        _, _, r = compute_fwd_rev_ratio(W_e_e[s:e, s:e], pref[s:e], SEQ_THETAS)
        ratios.append(r)
    return np.array(ratios)


def run_condition(label, n_hc, rf_spacing, seed, fixed_phases):
    print(f"\n{'='*76}")
    print(f"  {label}")
    print(f"  n_hc={n_hc}, M_per_hc={M_PER_HC}, rf={rf_spacing}, seed={seed}")
    print(f"  target_frac={TARGET_FRAC}, phases={'FIXED' if fixed_phases is not None else 'RANDOM'}")
    print(f"{'='*76}")

    t0 = time.perf_counter()

    p = Params(M=M_PER_HC, N=8, seed=seed, n_hc=n_hc,
               rf_spacing_pix=rf_spacing,
               ee_stdp_enabled=True, ee_connectivity="all_to_all",
               ee_stdp_A_plus=A_PLUS, ee_stdp_A_minus=A_MINUS,
               ee_stdp_weight_dep=True, train_segments=0, segment_ms=300.0)
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)

    # Phase A
    for seg in range(100):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)
    print(f"  Phase A: {time.perf_counter()-t0:.1f}s")

    # Tuning
    rates = evaluate_tuning_jax(state, static, THETAS_EVAL, repeats=2)
    osi, pref = compute_osi(rates, THETAS_EVAL)
    print(f"  Pre-cal OSI: mean={osi.mean():.3f}")

    if n_hc > 1:
        for hc in range(n_hc):
            s, e = hc * M_PER_HC, (hc + 1) * M_PER_HC
            cov = {}
            for th in SEQ_THETAS:
                d = np.abs(pref[s:e] - th)
                d = np.minimum(d, 180 - d)
                cov[th] = int(np.sum(d < 22.5))
            min_cov = min(cov.values())
            print(f"    HC{hc}: OSI={osi[s:e].mean():.3f}, coverage={cov}, min={min_cov}")

    # Calibrate with lower target_frac
    scale, _ = calibrate_ee_drive_jax(state, static, target_frac=TARGET_FRAC, osi_floor=0.30)
    state, static, _, _ = prepare_phaseb_ee(state, static, scale)
    print(f"  scale={scale:.1f}, w_max={float(static.w_e_e_max):.4f}")

    # Post-cal OSI
    rates_post = evaluate_tuning_jax(state, static, THETAS_EVAL, repeats=2)
    osi_post, _ = compute_osi(rates_post, THETAS_EVAL)
    print(f"  Post-cal OSI: mean={osi_post.mean():.3f}")
    print(f"  Setup: {time.perf_counter()-t0:.1f}s")

    # Phase B
    results = []
    cp_idx = 0
    t0b = time.perf_counter()

    for k in range(1, N_PRES + 1):
        kwargs = dict(plastic_mode='ee', ee_A_plus_eff=A_PLUS, ee_A_minus_eff=A_MINUS)
        if fixed_phases is not None:
            kwargs['phases'] = fixed_phases

        state, _ = run_sequence_trial_jax(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, 1.0, **kwargs)

        if cp_idx < len(CHECKPOINTS) and k == CHECKPOINTS[cp_idx]:
            W = get_flat_W_e_e_numpy(state, static)
            elapsed = time.perf_counter() - t0b

            if n_hc > 1:
                per_hc = compute_per_hc_fr(W, pref, n_hc, M_PER_HC)
                fr_med = float(np.median(per_hc))
                frac_gt1 = float(np.mean(per_hc > 1.0))
                hc_str = ' '.join(f'{v:.3f}' for v in per_hc)
                print(f"  [pres {k:4d}] med={fr_med:.4f} [{hc_str}] frac>1={frac_gt1:.0%} {elapsed:.1f}s")
                results.append({'pres': k, 'fr_median': fr_med, 'per_hc': per_hc.copy(), 'frac_gt1': frac_gt1})
            else:
                _, _, fr = compute_fwd_rev_ratio(W, pref, SEQ_THETAS)
                print(f"  [pres {k:4d}] F>R={fr:.4f} {elapsed:.1f}s")
                results.append({'pres': k, 'fr_median': fr, 'per_hc': np.array([fr]), 'frac_gt1': 1.0 if fr > 1.0 else 0.0})
            cp_idx += 1

    total = time.perf_counter() - t0b
    print(f"  Phase B: {total:.1f}s ({total/N_PRES*1000:.0f}ms/pres)")

    return results, float(osi.mean()), float(osi_post.mean())


def main():
    print("=" * 76)
    print("MULTI-HC PHASE B WITH target_frac=0.05")
    print(f"JAX devices: {jax.devices()}")
    print(f"M_per_hc={M_PER_HC}, target_frac={TARGET_FRAC}")
    print(f"Seeds: {SEEDS}")
    print("=" * 76)

    fixed_phases = jnp.array([0.0, 0.0, 0.0, 0.0])
    all_runs = {}

    for seed in SEEDS:
        # Condition A: n_hc=4, rf=1.0, fixed phases
        res_a, pre_osi_a, post_osi_a = run_condition(
            f"COND A: n_hc=4 rf=1.0 FIXED seed={seed}",
            N_HC, 1.0, seed, fixed_phases)
        all_runs[f'A_s{seed}'] = (res_a, pre_osi_a, post_osi_a)

    # One baseline: n_hc=1, seed=42
    res_b, pre_osi_b, post_osi_b = run_condition(
        "COND B: n_hc=1 rf=0 RANDOM seed=42 (baseline)",
        1, 0.0, 42, None)
    all_runs['B_baseline'] = (res_b, pre_osi_b, post_osi_b)

    # Summary
    print(f"\n{'='*76}")
    print("SUMMARY")
    print(f"{'='*76}")

    for name, (results, pre_osi, post_osi) in all_runs.items():
        traj = [(r['pres'], r['fr_median']) for r in results]
        final_fr = results[-1]['fr_median'] if results else 0
        final_frac = results[-1].get('frac_gt1', 0)
        print(f"\n  {name}: pre_OSI={pre_osi:.3f}, post_OSI={post_osi:.3f}")
        for p, fr in traj:
            print(f"    pres={p:4d}: F>R={fr:.4f}")
        print(f"    Final: F>R={final_fr:.4f}, frac>1={final_frac:.0%}")

        if 'A_' in name and len(results) > 0 and len(results[-1]['per_hc']) > 1:
            print(f"    Per-HC at {results[-1]['pres']}: "
                  + ' '.join(f'HC{i}={v:.3f}' for i, v in enumerate(results[-1]['per_hc'])))

    # Verdict
    print(f"\n{'='*76}")
    print("VERDICT")
    print(f"{'='*76}")

    for seed in SEEDS:
        key = f'A_s{seed}'
        res = all_runs[key][0]
        if res:
            fr = res[-1]['fr_median']
            frac = res[-1]['frac_gt1']
            all_gt1 = all(res[-1]['per_hc'] > 1.0) if len(res[-1]['per_hc']) > 1 else fr > 1.0
            pass_fr = fr > 1.10
            monotonic = all(res[i]['fr_median'] <= res[i+1]['fr_median'] for i in range(len(res)-1))
            print(f"  Seed {seed}: F>R_med={fr:.4f} {'PASS' if pass_fr else 'FAIL'}(>1.10) "
                  f"all_HCs>1={all_gt1} monotonic={monotonic} frac>1={frac:.0%}")

    key_b = 'B_baseline'
    res_b = all_runs[key_b][0]
    if res_b:
        fr_b = res_b[-1]['fr_median']
        print(f"  Baseline (n_hc=1): F>R={fr_b:.4f} {'PASS' if fr_b > 1.10 else 'FAIL'}(>1.10)")

    print("\nDone.")


if __name__ == "__main__":
    main()

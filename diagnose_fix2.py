#!/usr/bin/env python
"""diagnose_fix2.py — Focused test of fix candidates for 300-segment F>R.

Shares Phase A across conditions. Tests 400 presentations (enough to see trend).
"""
import sys, time, math
import numpy as np
sys.path.insert(0, '.')

from biologically_plausible_v1_stdp import Params, RgcLgnV1Network, compute_osi, calibrate_ee_drive
from network_jax import numpy_net_to_jax_state, run_segment_jax, run_sequence_trial_jax, evaluate_tuning_jax
import jax.numpy as jnp

GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
THETA_STEP = 180.0 / GOLDEN_RATIO
SEQ_THETAS = [0.0, 45.0, 90.0, 135.0]
ELEMENT_MS = 150.0
ITI_MS = 1500.0

def compute_fwd_rev_ratio(W_e_e, pref, seq_thetas):
    fwd_ws, rev_ws = [], []
    for ei in range(len(seq_thetas) - 1):
        pre_th, post_th = seq_thetas[ei], seq_thetas[ei + 1]
        d_pre = np.abs(pref - pre_th); d_pre = np.minimum(d_pre, 180.0 - d_pre)
        d_post = np.abs(pref - post_th); d_post = np.minimum(d_post, 180.0 - d_post)
        for pi in np.where(d_post < 22.5)[0]:
            for pj in np.where(d_pre < 22.5)[0]:
                if pi != pj:
                    fwd_ws.append(W_e_e[pi, pj]); rev_ws.append(W_e_e[pj, pi])
    if not fwd_ws: return 0, 0, 1, 0
    return float(np.mean(fwd_ws)), float(np.mean(rev_ws)), float(np.mean(fwd_ws))/max(1e-10, float(np.mean(rev_ws))), len(fwd_ws)

def main():
    seed = 42
    p = Params(M=16, N=8, seed=seed, ee_stdp_enabled=True, ee_connectivity="all_to_all",
               ee_stdp_A_plus=0.005, ee_stdp_A_minus=0.006,
               ee_stdp_weight_dep=True, train_segments=0, segment_ms=300.0)
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)

    print("Phase A (300 segments, JAX)...")
    t0 = time.perf_counter()
    for seg in range(300):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    # Pre-cal pref
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates_pre = evaluate_tuning_jax(state, static, thetas_eval, repeats=3)
    osi_pre, pref_pre = compute_osi(rates_pre, thetas_eval)
    print(f"  Pre-cal OSI={osi_pre.mean():.3f}")
    print(f"  Pre-cal prefs: {np.sort(pref_pre).round(1)}")

    # Save JAX FF weights for reuse
    W_ff_jax = np.array(state.W)

    conditions = [
        # (label, target_frac, w_max_mult, osi_floor)
        ("BASELINE: frac=0.15, 3x, osi=0.3", 0.15, 3.0, 0.3),
        ("FIX-A: frac=0.03, 3x, osi=0.3",    0.03, 3.0, 0.3),
        ("FIX-B: frac=0.05, 3x, osi=0.3",    0.05, 3.0, 0.3),
        ("FIX-C: frac=0.15, 3x, osi=0.6",    0.15, 3.0, 0.6),
        ("FIX-D: frac=0.15, 2x, osi=0.6",    0.15, 2.0, 0.6),
    ]

    all_results = {}
    for label, target_frac, w_max_mult, osi_floor in conditions:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        # Fresh network for calibration
        net2 = RgcLgnV1Network(p)
        net2.W = W_ff_jax.copy()  # reuse JAX-trained FF weights

        scale, frac = calibrate_ee_drive(net2, target_frac=target_frac, osi_floor=osi_floor)
        mask_ee = net2.mask_e_e.astype(bool)
        np.fill_diagonal(mask_ee, False)
        cal_mean = float(net2.W_e_e[mask_ee].mean())
        new_w_max = max(cal_mean * w_max_mult, net2.p.w_e_e_max)
        net2.p.w_e_e_max = new_w_max
        print(f"  scale={scale:.1f}, frac={frac:.4f}, cal_mean={cal_mean:.4f}, w_max={new_w_max:.4f}")

        st, sc = numpy_net_to_jax_state(net2)

        # Post-cal pref
        rates_post = evaluate_tuning_jax(st, sc, thetas_eval, repeats=3)
        osi_post, pref_post = compute_osi(rates_post, thetas_eval)
        print(f"  Post-cal OSI={osi_post.mean():.3f}")

        A_plus = float(sc.ee_stdp_A_plus)
        A_minus = float(sc.ee_stdp_A_minus)

        # Run Phase B with both pref metrics
        checkpoints = [0, 50, 100, 200, 400]
        n_pres = 400
        fr_pre_list = []
        fr_post_list = []

        W_ee = np.array(st.W_e_e)
        _, _, r_pre, np_pre = compute_fwd_rev_ratio(W_ee, pref_pre, SEQ_THETAS)
        _, _, r_post, np_post = compute_fwd_rev_ratio(W_ee, pref_post, SEQ_THETAS)
        fr_pre_list.append((0, r_pre))
        fr_post_list.append((0, r_post))
        print(f"  [pres   0] pre={r_pre:.4f}({np_pre}p) post={r_post:.4f}({np_post}p)")

        t0 = time.perf_counter()
        next_cp = 1
        for k in range(1, n_pres + 1):
            st, _ = run_sequence_trial_jax(
                st, sc, SEQ_THETAS, ELEMENT_MS, ITI_MS, 1.0,
                'ee', ee_A_plus_eff=A_plus, ee_A_minus_eff=A_minus)
            if next_cp < len(checkpoints) and k == checkpoints[next_cp]:
                W_ee = np.array(st.W_e_e)
                off = W_ee[mask_ee]
                _, _, r_pre, _ = compute_fwd_rev_ratio(W_ee, pref_pre, SEQ_THETAS)
                _, _, r_post, _ = compute_fwd_rev_ratio(W_ee, pref_post, SEQ_THETAS)
                fr_pre_list.append((k, r_pre))
                fr_post_list.append((k, r_post))
                print(f"  [pres {k:3d}] pre={r_pre:.4f} post={r_post:.4f} "
                      f"W:mean={off.mean():.4f} max={off.max():.4f} [{time.perf_counter()-t0:.1f}s]")
                next_cp += 1

        all_results[label] = (fr_pre_list, fr_post_list)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY (F>R with pre-calibration pref / post-calibration pref)")
    print("=" * 80)
    for label, (fr_pre, fr_post) in all_results.items():
        traj_pre = ' → '.join(f'{r:.3f}' for _, r in fr_pre)
        traj_post = ' → '.join(f'{r:.3f}' for _, r in fr_post)
        final_pre = fr_pre[-1][1]
        final_post = fr_post[-1][1]
        s_pre = "OK" if final_pre > 1.0 else "FAIL"
        s_post = "OK" if final_post > 1.0 else "FAIL"
        print(f"\n  {label}")
        print(f"    pre-pref:  {traj_pre}  [{s_pre}]")
        print(f"    post-pref: {traj_post}  [{s_post}]")

    print("\n[DONE]")

if __name__ == '__main__':
    main()

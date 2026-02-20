#!/usr/bin/env python3
"""compare_before_after.py - Generate before/after visualizations for the PV+STDP fix.

Produces:
  1. Smoothed feedforward weights (before/after Phase A — same for both, shown once)
  2. Preference histograms (same for both)
  3. 7 sequence learning figures for "BEFORE" (broken: weight-dep STDP, calibrated W_e_e)
  4. 7 sequence learning figures for "AFTER"  (fixed: additive STDP, W_e_e/W_pv_e reset)

Phase A is shared (run once). Phase B runs twice with different configs.
Root cause: Phase A iSTDP equilibrium leaves W_pv_e ~ 0.02 (local PV), but Phase B
resets W_e_e to 0.01 → PV inhibition > recurrent excitation → E firing suppressed
→ STDP cannot build forward/reverse asymmetry.
"""
import os
import sys
import math
import time
import numpy as np
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from biologically_plausible_v1_stdp import (
    Params, RgcLgnV1Network, compute_osi,
    run_sequence_trial, calibrate_ee_drive,
    plot_weight_maps_before_after,
    plot_pref_hist,
    plot_osi_development,
    plot_forward_reverse_asymmetry,
    plot_ee_weight_matrix_evolution,
    plot_omission_prediction_growth,
    plot_omission_activity_traces,
    plot_omission_traces_evolution,
    plot_full_sequence_response_evolution,
    plot_sequence_distance_analysis,
)

# ======================== Constants ========================
SEED = 42
PHASE_A_SEGS = 300
SEQ_THETAS = [5.0, 65.0, 125.0]
GROUP_WINDOW = 28.0
SEQ_CONTRAST = 2.0
ELEMENT_MS = 30.0
ITI_MS = 200.0
TOTAL_PRES = 800
CKPT_INTERVAL = 200
OMIT_INDEX = 1

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_before_after")
os.makedirs(OUT, exist_ok=True)

# ======================== Helpers ========================

def fwd_rev_asymmetry(net, seq_thetas, pref, gw=GROUP_WINDOW):
    fwd_ws, rev_ws = [], []
    for ei in range(len(seq_thetas) - 1):
        d1 = np.minimum(np.abs(pref - seq_thetas[ei]), 180.0 - np.abs(pref - seq_thetas[ei]))
        d2 = np.minimum(np.abs(pref - seq_thetas[ei+1]), 180.0 - np.abs(pref - seq_thetas[ei+1]))
        for pi in np.where(d2 < gw)[0]:
            for pj in np.where(d1 < gw)[0]:
                if pi != pj:
                    fwd_ws.append(float(net.W_e_e[pi, pj]))
                    rev_ws.append(float(net.W_e_e[pj, pi]))
    if not fwd_ws:
        return 0.0, 0.0, 1.0
    return float(np.mean(fwd_ws)), float(np.mean(rev_ws)), \
           float(np.mean(fwd_ws)) / max(1e-10, float(np.mean(rev_ws)))


def weight_prediction(net, omit_mask, pre_mask, ctrl_mask):
    fwd_w = float(net.W_e_e[np.ix_(omit_mask, pre_mask)].mean())
    ctrl_w = float(net.W_e_e[np.ix_(omit_mask, ctrl_mask)].mean())
    return fwd_w, ctrl_w, fwd_w - ctrl_w


def collect_traces(net, p, pref, omit_mask, pre_mask, ctrl_mask, ctrl_thetas, n_avg=15):
    n_pre = int(pre_mask.sum())
    n_ctrl = int(ctrl_mask.sum())
    snap = net.save_dynamic_state()
    was_active = net.ee_stdp_active
    net.ee_stdp_active = False
    oi = OMIT_INDEX
    dt = p.dt_ms
    omit_start = int(oi * ELEMENT_MS / dt)
    omit_end = int((oi + 1) * ELEMENT_MS / dt)

    om_traces, ct_traces, fs_traces = [], [], []
    diff_trials = []
    for _ in range(n_avg):
        net.vep_target_mask = omit_mask
        net.reset_state()
        om = run_sequence_trial(net, SEQ_THETAS, ELEMENT_MS, ITI_MS, SEQ_CONTRAST,
                                plastic=False, omit_index=oi, record=True, vep_mode="g_exc_ee")
        om_full = np.concatenate(om["element_traces"] + [om["iti_trace"]])
        om_traces.append(om_full / max(n_pre, 1))

        net.reset_state()
        ct = run_sequence_trial(net, ctrl_thetas, ELEMENT_MS, ITI_MS, SEQ_CONTRAST,
                                plastic=False, omit_index=oi, record=True, vep_mode="g_exc_ee")
        ct_full = np.concatenate(ct["element_traces"] + [ct["iti_trace"]])
        ct_traces.append(ct_full / max(n_ctrl, 1))

        om_win = float(np.mean(om_full[omit_start:omit_end]))
        ct_win = float(np.mean(ct_full[omit_start:omit_end]))
        diff_trials.append(om_win - ct_win)

        net.vep_target_mask = None
        net.reset_state()
        fs = run_sequence_trial(net, SEQ_THETAS, ELEMENT_MS, ITI_MS, SEQ_CONTRAST,
                                plastic=False, record=True, vep_mode="g_exc_ee")
        fs_traces.append(np.concatenate(fs["element_traces"] + [fs["iti_trace"]]))

    net.vep_target_mask = None
    net.ee_stdp_active = was_active
    net.restore_dynamic_state(snap)
    return (np.mean(om_traces, axis=0), np.mean(ct_traces, axis=0),
            np.mean(fs_traces, axis=0), float(np.mean(diff_trials)), np.array(diff_trials))


def save_checkpoint(net):
    return {
        'W': net.W.copy(),
        'W_e_e': net.W_e_e.copy(), 'W_pv_e': net.W_pv_e.copy(),
        'dyn': net.save_dynamic_state(),
    }


def restore_checkpoint(net, ckpt):
    net.W[:] = ckpt['W']
    net.W_e_e[:] = ckpt['W_e_e']
    net.W_pv_e[:] = ckpt['W_pv_e']
    net.restore_dynamic_state(ckpt['dyn'])


def run_phase_b(net, p, pref, label, a_plus, a_minus, weight_dep=True,
                 reset_wee=None, reset_wpve=False):
    """Run full Phase B and return all data needed for the 7 figures.

    Args:
        reset_wee: if not None, reset W_e_e to this value (e.g. 0.01)
        reset_wpve: if True, reset W_pv_e to 0 and pv_istdp
    """
    # Set STDP parameters
    p.ee_stdp_A_plus = a_plus
    p.ee_stdp_A_minus = a_minus
    p.ee_stdp_weight_dep = weight_dep

    # Optional weight resets for additive STDP
    if reset_wee is not None:
        net.W_e_e[:] = reset_wee
        np.fill_diagonal(net.W_e_e, 0.0)
        net.W_e_e *= net.mask_e_e.astype(np.float32)
        p.w_e_e_max = 0.2
        print(f"  [{label}] Reset W_e_e={reset_wee}, w_max={p.w_e_e_max}")
    if reset_wpve:
        net.W_pv_e[:] = 0.0
        net.pv_istdp.reset()
        print(f"  [{label}] Reset W_pv_e=0, pv_istdp reset")

    net.ff_plastic_enabled = False
    net.ee_stdp_active = True
    net._ee_stdp_ramp_factor = 1.0
    net.reset_state()
    net.delay_ee_stdp.reset()

    # Setup omission/prediction groups
    omit_theta = SEQ_THETAS[OMIT_INDEX]
    pre_omit_theta = SEQ_THETAS[OMIT_INDEX - 1]
    ctrl_pre_theta = float(SEQ_THETAS[-1])

    d_omit = np.minimum(np.abs(pref - omit_theta), 180.0 - np.abs(pref - omit_theta))
    omit_mask = d_omit < GROUP_WINDOW
    d_pre = np.minimum(np.abs(pref - pre_omit_theta), 180.0 - np.abs(pref - pre_omit_theta))
    pre_mask = d_pre < GROUP_WINDOW
    d_ctrl = np.minimum(np.abs(pref - ctrl_pre_theta), 180.0 - np.abs(pref - ctrl_pre_theta))
    ctrl_mask = d_ctrl < GROUP_WINDOW

    ctrl_thetas = list(SEQ_THETAS)
    ctrl_thetas[OMIT_INDEX - 1] = ctrl_pre_theta

    print(f"  [{label}] Groups: omit={omit_mask.sum()}, pre={pre_mask.sum()}, ctrl={ctrl_mask.sum()}")

    # Baseline
    fwd_0, rev_0, ratio_0 = fwd_rev_asymmetry(net, SEQ_THETAS, pref)
    wfwd_0, wctrl_0, wpred_0 = weight_prediction(net, omit_mask, pre_mask, ctrl_mask)
    om_0, ct_0, fs_0, spred_0, diff_0 = collect_traces(
        net, p, pref, omit_mask, pre_mask, ctrl_mask, ctrl_thetas)

    ckpt_pres = [0]
    all_ratios = [ratio_0]
    all_wpreds = [wpred_0]
    all_spreds = [spred_0]
    fine_pres = [0]
    fine_fwds = [fwd_0]
    fine_revs = [rev_0]
    fine_ratios = [ratio_0]
    fine_wfwds = [wfwd_0]
    fine_wctrls = [wctrl_0]
    fine_wpreds = [wpred_0]
    W_snapshots = {0: net.W_e_e.copy()}
    omission_traces = {0: om_0}
    control_traces = {0: ct_0}
    full_seq_traces = {0: fs_0}
    diff_trials_dict = {0: diff_0}

    print(f"  [{label}] Pre: ratio={ratio_0:.3f}, W_pred={wpred_0:.5f}")

    for k in range(1, TOTAL_PRES + 1):
        run_sequence_trial(net, SEQ_THETAS, ELEMENT_MS, ITI_MS, SEQ_CONTRAST,
                           plastic=True, vep_mode="spikes")
        if k % 25 == 0:
            fwd_k, rev_k, ratio_k = fwd_rev_asymmetry(net, SEQ_THETAS, pref)
            wfwd_k, wctrl_k, wpred_k = weight_prediction(net, omit_mask, pre_mask, ctrl_mask)
            fine_pres.append(k)
            fine_fwds.append(fwd_k)
            fine_revs.append(rev_k)
            fine_ratios.append(ratio_k)
            fine_wfwds.append(wfwd_k)
            fine_wctrls.append(wctrl_k)
            fine_wpreds.append(wpred_k)
            if k % 100 == 0:
                print(f"  [{label} pres {k}/{TOTAL_PRES}] ratio={ratio_k:.3f}, W_pred={wpred_k:.5f}")
        if k % CKPT_INTERVAL == 0:
            fwd_k, rev_k, ratio_k = fwd_rev_asymmetry(net, SEQ_THETAS, pref)
            wfwd_k, wctrl_k, wpred_k = weight_prediction(net, omit_mask, pre_mask, ctrl_mask)
            om_k, ct_k, fs_k, spred_k, diff_k = collect_traces(
                net, p, pref, omit_mask, pre_mask, ctrl_mask, ctrl_thetas)
            ckpt_pres.append(k)
            all_ratios.append(ratio_k)
            all_wpreds.append(wpred_k)
            all_spreds.append(spred_k)
            W_snapshots[k] = net.W_e_e.copy()
            omission_traces[k] = om_k
            control_traces[k] = ct_k
            full_seq_traces[k] = fs_k
            diff_trials_dict[k] = diff_k
            print(f"  [{label} ckpt {k}] ratio={ratio_k:.3f}, W_pred={wpred_k:.5f}, g_ee_pred={spred_k:.5f}")

    return {
        'ckpt_pres': ckpt_pres, 'all_ratios': all_ratios, 'all_wpreds': all_wpreds,
        'all_spreds': all_spreds,
        'fine_pres': fine_pres, 'fine_fwds': fine_fwds, 'fine_revs': fine_revs,
        'fine_ratios': fine_ratios, 'fine_wfwds': fine_wfwds, 'fine_wctrls': fine_wctrls,
        'fine_wpreds': fine_wpreds,
        'W_snapshots': W_snapshots, 'omission_traces': omission_traces,
        'control_traces': control_traces, 'full_seq_traces': full_seq_traces,
        'diff_trials_dict': diff_trials_dict,
    }


def generate_7_figures(data, pref, dt_ms, out_dir, label):
    """Generate all 7 sequence learning figures from Phase B data."""
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, label)

    plot_forward_reverse_asymmetry(
        data['fine_pres'], data['fine_fwds'], data['fine_revs'], data['fine_ratios'],
        data['ckpt_pres'], list(data['all_ratios']),
        f"{prefix}_fig2_fwd_rev.png")

    plot_ee_weight_matrix_evolution(
        data['W_snapshots'], pref, SEQ_THETAS,
        f"{prefix}_fig3_weight_matrix.png", group_window=GROUP_WINDOW)

    plot_omission_prediction_growth(
        data['fine_pres'], data['fine_wfwds'], data['fine_wctrls'], data['fine_wpreds'],
        data['ckpt_pres'], list(data['all_wpreds']),
        data['diff_trials_dict'],
        f"{prefix}_fig4_prediction_growth.png")

    plot_omission_activity_traces(
        data['omission_traces'], data['control_traces'],
        data['diff_trials_dict'],
        ELEMENT_MS, ITI_MS, SEQ_THETAS, OMIT_INDEX, dt_ms,
        f"{prefix}_fig5_omission_traces.png")

    plot_omission_traces_evolution(
        data['omission_traces'], data['control_traces'],
        ELEMENT_MS, ITI_MS, SEQ_THETAS, OMIT_INDEX, dt_ms,
        f"{prefix}_fig5b_traces_evolution.png")

    plot_full_sequence_response_evolution(
        data['full_seq_traces'], ELEMENT_MS, ITI_MS, SEQ_THETAS, dt_ms,
        f"{prefix}_fig6_full_seq_response.png")

    plot_sequence_distance_analysis(
        data['W_snapshots'], pref, SEQ_THETAS,
        f"{prefix}_fig7_distance_analysis.png", group_window=GROUP_WINDOW)

    print(f"  [{label}] 7 figures saved to {out_dir}")


# ======================== Main ========================

def main():
    t0 = time.time()
    print("=" * 70)
    print("BEFORE/AFTER COMPARISON: PV + STDP Fix")
    print("=" * 70)

    # ==================================================
    # Shared Phase A
    # ==================================================
    print("\n[Phase A] Building network and training FF STDP...")
    # STDP params set per condition in run_phase_b; placeholders here
    p = Params(
        N=8, M=32, seed=SEED,
        train_segments=0, segment_ms=300,
        train_contrast=SEQ_CONTRAST,
        v1_bias_init=0.0,
        ee_stdp_enabled=True,
        ee_connectivity="all_to_all",
        ee_stdp_A_plus=0.005,     # placeholder, overridden per condition
        ee_stdp_A_minus=0.006,
        ee_stdp_weight_dep=True,
        rgc_separate_onoff_mosaics=True,
    )
    net = RgcLgnV1Network(p, init_mode="random")

    # Save initial weights for before/after weight maps
    W_init = net.W.copy()

    # Phase A: FF STDP
    net.ff_plastic_enabled = True
    net.ee_stdp_active = False
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    theta_step = 180.0 / phi
    theta_offset = float(net.rng.uniform(0.0, 180.0))

    phaseA_osi_segs: List[int] = []
    phaseA_osi_means: List[float] = []
    phaseA_osi_stds: List[float] = []
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)

    for s in range(1, PHASE_A_SEGS + 1):
        th = float((theta_offset + (s - 1) * theta_step) % 180.0)
        net.run_segment(th, plastic=True, contrast=SEQ_CONTRAST)
        if s % 25 == 0:
            _rates = net.evaluate_tuning(thetas_eval, repeats=3, contrast=SEQ_CONTRAST)
            _osi, _ = compute_osi(_rates, thetas_eval)
            phaseA_osi_segs.append(s)
            phaseA_osi_means.append(float(_osi.mean()))
            phaseA_osi_stds.append(float(_osi.std()))
            if s % 100 == 0:
                print(f"  [Phase A seg {s}] OSI={_osi.mean():.3f}")

    rates = net.evaluate_tuning(thetas_eval, repeats=5, contrast=SEQ_CONTRAST)
    osi, pref = compute_osi(rates, thetas_eval)
    print(f"  [Phase A done] OSI={osi.mean():.3f}")

    # Calibrate E->E
    print("  Calibrating E->E...")
    net.ff_plastic_enabled = False
    net.ee_stdp_active = True
    net._ee_stdp_ramp_factor = 1.0
    cal_scale, cal_frac = calibrate_ee_drive(net, 0.15, osi_floor=0.1, contrast=SEQ_CONTRAST)
    cal_mean = float(net.W_e_e[net.mask_e_e].mean())
    p.w_e_e_max = max(cal_mean * 2.0, p.w_e_e_max)
    print(f"  [Cal] scale={cal_scale:.1f}, frac={cal_frac:.4f}, W_ee={cal_mean:.5f}")

    # Save post-Phase-A state
    ckpt = save_checkpoint(net)
    W_final = net.W.copy()

    # ==================================================
    # Phase A Visualizations (shared)
    # ==================================================
    print("\n[Viz] Generating Phase A visualizations...")

    # Fig 0a: Smoothed weight maps (before/after Phase A)
    plot_weight_maps_before_after(
        W_init, W_final, p.N,
        os.path.join(OUT, "phaseA_smoothed_weights.png"),
        "Feedforward Weights: Initial vs After Phase A",
        smooth_sigma=0.8)
    print("  phaseA_smoothed_weights.png")

    # Fig 0b: Preference histogram
    plot_pref_hist(
        pref, osi,
        os.path.join(OUT, "phaseA_pref_hist.png"),
        f"Preferred Orientations After Phase A (mean OSI={osi.mean():.3f})")
    print("  phaseA_pref_hist.png")

    # Fig 1: OSI Development (shared)
    plot_osi_development(
        phaseA_osi_segs, phaseA_osi_means, phaseA_osi_stds,
        osi, pref,
        os.path.join(OUT, "fig1_osi_development.png"))
    print("  fig1_osi_development.png")

    # ==================================================
    # BEFORE: weight-dep STDP with calibrated W_e_e (broken: universal depression)
    # ==================================================
    print(f"\n{'='*70}")
    print("[BEFORE] Weight-dep STDP, calibrated W_e_e (A+=0.005, A-=0.006)")
    print(f"{'='*70}")
    restore_checkpoint(net, ckpt)
    data_before = run_phase_b(net, p, pref, "BEFORE",
                              a_plus=0.005, a_minus=0.006, weight_dep=True)
    generate_7_figures(data_before, pref, p.dt_ms, OUT, "BEFORE")
    t_before = time.time()
    print(f"  [BEFORE done in {t_before - t0:.0f}s]")

    # ==================================================
    # AFTER: additive STDP + W_e_e/W_pv_e reset (fixes PV/E-E scale mismatch)
    # ==================================================
    print(f"\n{'='*70}")
    print("[AFTER] Additive STDP, W_e_e=0.01, W_pv_e=0 (A+=0.002, A-=0.0024)")
    print(f"{'='*70}")
    restore_checkpoint(net, ckpt)
    data_after = run_phase_b(net, p, pref, "AFTER",
                             a_plus=0.002, a_minus=0.0024, weight_dep=False,
                             reset_wee=0.01, reset_wpve=True)
    generate_7_figures(data_after, pref, p.dt_ms, OUT, "AFTER")
    t_after = time.time()
    print(f"  [AFTER done in {t_after - t_before:.0f}s]")

    # ==================================================
    # Comparison Summary Figure
    # ==================================================
    print("\n[Viz] Generating comparison summary...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Forward/reverse ratio
    ax = axes[0, 0]
    ax.plot(data_before['fine_pres'], data_before['fine_ratios'], 'r-', alpha=0.7, label='Before (wt-dep STDP)')
    ax.plot(data_after['fine_pres'], data_after['fine_ratios'], 'b-', alpha=0.7, label='After (additive+reset)')
    ax.axhline(1.5, color='green', linestyle='--', alpha=0.5, label='Target (1.5)')
    ax.set_xlabel('Presentations')
    ax.set_ylabel('Forward / Reverse ratio')
    ax.set_title('Sequence Learning: Forward/Reverse Asymmetry')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Mean W_e_e
    b_wee = [float(data_before['W_snapshots'][k][net.mask_e_e].mean()) for k in sorted(data_before['W_snapshots'])]
    a_wee = [float(data_after['W_snapshots'][k][net.mask_e_e].mean()) for k in sorted(data_after['W_snapshots'])]
    ax = axes[0, 1]
    ax.plot(data_before['ckpt_pres'], b_wee, 'ro-', label='Before')
    ax.plot(data_after['ckpt_pres'], a_wee, 'bs-', label='After')
    ax.set_xlabel('Presentations')
    ax.set_ylabel('Mean W_e_e')
    ax.set_title('E-E Weight Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Weight-based prediction
    ax = axes[1, 0]
    ax.plot(data_before['fine_pres'], data_before['fine_wpreds'], 'r-', alpha=0.7, label='Before')
    ax.plot(data_after['fine_pres'], data_after['fine_wpreds'], 'b-', alpha=0.7, label='After')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Presentations')
    ax.set_ylabel('W_pred (fwd - ctrl)')
    ax.set_title('Weight-Based Omission Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Conductance-based prediction
    ax = axes[1, 1]
    ax.plot(data_before['ckpt_pres'], data_before['all_spreds'], 'ro-', label='Before')
    ax.plot(data_after['ckpt_pres'], data_after['all_spreds'], 'bs-', label='After')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Presentations')
    ax.set_ylabel('g_exc_ee prediction (omit - ctrl)')
    ax.set_title('Conductance-Based Omission Prediction (VEP analog)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Before vs After: PV/E-E Scale Mismatch Fix\n'
                 'Before: wt-dep STDP, calibrated W_e_e  |  '
                 'After: additive STDP, W_e_e=0.01, W_pv_e=0', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "comparison_summary.png"), dpi=150)
    plt.close(fig)
    print("  comparison_summary.png")

    # ==================================================
    # Summary
    # ==================================================
    t_end = time.time()
    print(f"\n{'='*70}")
    print(f"COMPARISON COMPLETE ({t_end - t0:.0f}s)")
    print(f"{'='*70}")
    b_final = data_before['all_ratios'][-1]
    a_final = data_after['all_ratios'][-1]
    print(f"  BEFORE: final ratio={b_final:.3f}, W_pred={data_before['all_wpreds'][-1]:.5f}, "
          f"g_pred={data_before['all_spreds'][-1]:.5f}")
    print(f"  AFTER:  final ratio={a_final:.3f}, W_pred={data_after['all_wpreds'][-1]:.5f}, "
          f"g_pred={data_after['all_spreds'][-1]:.5f}")
    print(f"  Improvement: ratio {b_final:.3f} -> {a_final:.3f} ({a_final/max(b_final,0.001):.1f}x)")
    print(f"\n  All figures saved to: {OUT}")


if __name__ == "__main__":
    main()

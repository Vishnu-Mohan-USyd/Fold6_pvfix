#!/usr/bin/env python3
"""Generate the 7 sequence learning publication figures.

Standalone script that replicates the selftest's data collection
using additive STDP with W_e_e/W_pv_e reset for clean Phase B
learning (avoiding PV/E-E scale mismatch from Phase A iSTDP equilibrium).
"""
import os
import sys
import math
import numpy as np
from typing import List

import matplotlib
matplotlib.use("Agg")

from biologically_plausible_v1_stdp import (
    Params, RgcLgnV1Network, compute_osi,
    run_sequence_trial, calibrate_ee_drive,
    plot_osi_development,
    plot_forward_reverse_asymmetry,
    plot_ee_weight_matrix_evolution,
    plot_omission_prediction_growth,
    plot_omission_activity_traces,
    plot_omission_traces_evolution,
    plot_full_sequence_response_evolution,
    plot_sequence_distance_analysis,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_sequence_learning")
os.makedirs(OUT, exist_ok=True)

SEED = 42
PHASE_A_SEGS = 300
SEQ_THETAS = [5.0, 65.0, 125.0]  # 3 elements, 60 deg apart
GROUP_WINDOW = 28.0
SEQ_CONTRAST = 2.0
ELEMENT_MS = 30.0
ITI_MS = 200.0
TOTAL_PRES = 800
CKPT_INTERVAL = 200
OMIT_INDEX = 1

# --- Build network with additive STDP (matching self-test protocol) ---
print("[seq-figs] Building network...")
p = Params(
    N=8, M=32, seed=SEED,
    train_segments=0, segment_ms=300,
    train_contrast=SEQ_CONTRAST,
    v1_bias_init=0.0,
    ee_stdp_enabled=True,
    ee_connectivity="all_to_all",
    ee_stdp_A_plus=0.002,
    ee_stdp_A_minus=0.0024,
    ee_stdp_weight_dep=False,
    rgc_separate_onoff_mosaics=True,
)
net = RgcLgnV1Network(p, init_mode="random")

# --- Phase A: feedforward STDP with OSI tracking ---
print(f"[seq-figs] Phase A: {PHASE_A_SEGS} segments...")
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
            print(f"  [Phase A seg {s}/{PHASE_A_SEGS}] mean OSI={_osi.mean():.3f}")

# Post Phase A evaluation
rates_shared = net.evaluate_tuning(thetas_eval, repeats=5, contrast=SEQ_CONTRAST)
osi_shared, pref_shared = compute_osi(rates_shared, thetas_eval)
print(f"  [Phase A complete] mean OSI={osi_shared.mean():.3f}")

# --- Calibrate E->E ---
print("[seq-figs] Calibrating E->E drive...")
net.ff_plastic_enabled = False
net.ee_stdp_active = True
net._ee_stdp_ramp_factor = 1.0

cal_scale, cal_frac = calibrate_ee_drive(net, 0.15, osi_floor=0.1, contrast=SEQ_CONTRAST)
cal_mean = float(net.W_e_e[net.mask_e_e].mean())
print(f"  [Calibration] scale={cal_scale:.1f}, frac={cal_frac:.4f}, W_ee mean={cal_mean:.5f}")

# --- Reset for additive STDP Phase B ---
# Phase A iSTDP equilibrium leaves W_pv_e ~ 0.02 (local PV sees driven rates
# > target_rate_hz=8).  With W_e_e reset to 0.01, PV inhibition is ~2x larger
# than recurrent excitation, suppressing E firing so severely that STDP cannot
# build forward/reverse asymmetry.  Resetting W_pv_e to 0 lets iSTDP rebuild
# PV inhibition from scratch, properly tracking the growing W_e_e.
print("[seq-figs] Resetting W_e_e/W_pv_e for Phase B...")
net.W_e_e[:] = 0.01
np.fill_diagonal(net.W_e_e, 0.0)
net.W_e_e *= net.mask_e_e.astype(np.float32)
p.w_e_e_max = 0.2
net.W_pv_e[:] = 0.0
net.pv_istdp.reset()
net.reset_state()
net.delay_ee_stdp.reset()
print(f"  W_e_e={net.W_e_e[net.mask_e_e].mean():.4f}, W_pv_e={net.W_pv_e.mean():.4f}, w_max={p.w_e_e_max}")

# --- Setup sequence learning ---
# Identify neuron groups
omit_theta = SEQ_THETAS[OMIT_INDEX]
pre_omit_theta = SEQ_THETAS[OMIT_INDEX - 1]
ctrl_pre_theta = float(SEQ_THETAS[-1])

d_omit = np.minimum(np.abs(pref_shared - omit_theta), 180.0 - np.abs(pref_shared - omit_theta))
omit_mask = d_omit < GROUP_WINDOW

d_pre = np.minimum(np.abs(pref_shared - pre_omit_theta), 180.0 - np.abs(pref_shared - pre_omit_theta))
pre_mask = d_pre < GROUP_WINDOW

d_ctrl = np.minimum(np.abs(pref_shared - ctrl_pre_theta), 180.0 - np.abs(pref_shared - ctrl_pre_theta))
ctrl_mask = d_ctrl < GROUP_WINDOW

n_omit = int(omit_mask.sum())
n_pre = int(pre_mask.sum())
n_ctrl = int(ctrl_mask.sum())
print(f"  [Setup] omit ({omit_theta} deg): {n_omit}, pre ({pre_omit_theta} deg): {n_pre}, ctrl ({ctrl_pre_theta} deg): {n_ctrl}")

ctrl_thetas = list(SEQ_THETAS)
ctrl_thetas[OMIT_INDEX - 1] = ctrl_pre_theta

# --- Helper functions ---
def fwd_rev_asymmetry():
    fwd_ws, rev_ws = [], []
    for ei in range(len(SEQ_THETAS) - 1):
        d1 = np.minimum(np.abs(pref_shared - SEQ_THETAS[ei]), 180.0 - np.abs(pref_shared - SEQ_THETAS[ei]))
        d2 = np.minimum(np.abs(pref_shared - SEQ_THETAS[ei+1]), 180.0 - np.abs(pref_shared - SEQ_THETAS[ei+1]))
        for pi in np.where(d2 < GROUP_WINDOW)[0]:
            for pj in np.where(d1 < GROUP_WINDOW)[0]:
                if pi != pj:
                    fwd_ws.append(float(net.W_e_e[pi, pj]))
                    rev_ws.append(float(net.W_e_e[pj, pi]))
    if len(fwd_ws) == 0:
        return 0.0, 0.0, 1.0
    return float(np.mean(fwd_ws)), float(np.mean(rev_ws)), \
           float(np.mean(fwd_ws)) / max(1e-10, float(np.mean(rev_ws)))

def weight_prediction():
    fwd_w = float(net.W_e_e[np.ix_(omit_mask, pre_mask)].mean())
    ctrl_w = float(net.W_e_e[np.ix_(omit_mask, ctrl_mask)].mean())
    return fwd_w, ctrl_w, fwd_w - ctrl_w

def collect_traces(n_avg=15):
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
    return np.mean(om_traces, axis=0), np.mean(ct_traces, axis=0), np.mean(fs_traces, axis=0), \
           float(np.mean(diff_trials)), np.array(diff_trials)

# --- Baseline ---
print("[seq-figs] Baseline measurements...")
fwd_0, rev_0, ratio_0 = fwd_rev_asymmetry()
wfwd_0, wctrl_0, wpred_0 = weight_prediction()
om_0, ct_0, fs_0, spred_0, diff_0 = collect_traces()
print(f"  [Pre] ratio={ratio_0:.3f}, W_pred={wpred_0:.5f}")

# Data accumulators
ckpt_pres_list = [0]
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

# --- Phase B: sequence training ---
print(f"[seq-figs] Phase B: {TOTAL_PRES} presentations...")
for k in range(1, TOTAL_PRES + 1):
    run_sequence_trial(net, SEQ_THETAS, ELEMENT_MS, ITI_MS, SEQ_CONTRAST,
                       plastic=True, vep_mode="spikes")
    if k % 25 == 0:
        fwd_k, rev_k, ratio_k = fwd_rev_asymmetry()
        wfwd_k, wctrl_k, wpred_k = weight_prediction()
        fine_pres.append(k)
        fine_fwds.append(fwd_k)
        fine_revs.append(rev_k)
        fine_ratios.append(ratio_k)
        fine_wfwds.append(wfwd_k)
        fine_wctrls.append(wctrl_k)
        fine_wpreds.append(wpred_k)
        if k % 50 == 0:
            print(f"  [pres {k}/{TOTAL_PRES}] ratio={ratio_k:.3f}, W_pred={wpred_k:.5f}")
    if k % CKPT_INTERVAL == 0:
        fwd_k, rev_k, ratio_k = fwd_rev_asymmetry()
        wfwd_k, wctrl_k, wpred_k = weight_prediction()
        print(f"  [Checkpoint {k}] Collecting traces...")
        om_k, ct_k, fs_k, spred_k, diff_k = collect_traces()
        ckpt_pres_list.append(k)
        all_ratios.append(ratio_k)
        all_wpreds.append(wpred_k)
        all_spreds.append(spred_k)
        W_snapshots[k] = net.W_e_e.copy()
        omission_traces[k] = om_k
        control_traces[k] = ct_k
        full_seq_traces[k] = fs_k
        diff_trials_dict[k] = diff_k
        print(f"    ratio={ratio_k:.3f}, W_pred={wpred_k:.5f}, g_ee_pred={spred_k:.5f}")

# --- Generate 7 figures ---
print("[seq-figs] Generating figures...")

# Fig 1: OSI Development
plot_osi_development(
    phaseA_osi_segs, phaseA_osi_means, phaseA_osi_stds,
    osi_shared, pref_shared,
    os.path.join(OUT, "fig1_osi_development.png"))
print("  fig1_osi_development.png")

# Fig 2: Forward/Reverse Asymmetry
plot_forward_reverse_asymmetry(
    fine_pres, fine_fwds, fine_revs, fine_ratios,
    ckpt_pres_list, list(all_ratios),
    os.path.join(OUT, "fig2_forward_reverse_asymmetry.png"))
print("  fig2_forward_reverse_asymmetry.png")

# Fig 3: Weight Matrix Evolution
plot_ee_weight_matrix_evolution(
    W_snapshots, pref_shared, SEQ_THETAS,
    os.path.join(OUT, "fig3_weight_matrix_evolution.png"),
    group_window=GROUP_WINDOW)
print("  fig3_weight_matrix_evolution.png")

# Fig 4: Omission Prediction Growth
plot_omission_prediction_growth(
    fine_pres, fine_wfwds, fine_wctrls, fine_wpreds,
    ckpt_pres_list, list(all_wpreds),
    diff_trials_dict,
    os.path.join(OUT, "fig4_omission_prediction_growth.png"))
print("  fig4_omission_prediction_growth.png")

# Fig 5: Omission Activity Traces
plot_omission_activity_traces(
    omission_traces, control_traces,
    diff_trials_dict,
    ELEMENT_MS, ITI_MS, SEQ_THETAS, OMIT_INDEX, p.dt_ms,
    os.path.join(OUT, "fig5_omission_activity_traces.png"))
print("  fig5_omission_activity_traces.png")

# Fig 5b: Omission Traces Evolution
plot_omission_traces_evolution(
    omission_traces, control_traces,
    ELEMENT_MS, ITI_MS, SEQ_THETAS, OMIT_INDEX, p.dt_ms,
    os.path.join(OUT, "fig5b_omission_traces_evolution.png"))
print("  fig5b_omission_traces_evolution.png")

# Fig 6: Full Sequence Response Evolution
plot_full_sequence_response_evolution(
    full_seq_traces, ELEMENT_MS, ITI_MS, SEQ_THETAS, p.dt_ms,
    os.path.join(OUT, "fig6_full_sequence_response_evolution.png"))
print("  fig6_full_sequence_response_evolution.png")

# Fig 7: Sequence Distance Analysis
plot_sequence_distance_analysis(
    W_snapshots, pref_shared, SEQ_THETAS,
    os.path.join(OUT, "fig7_sequence_distance_analysis.png"),
    group_window=GROUP_WINDOW)
print("  fig7_sequence_distance_analysis.png")

# Save data
np.savez_compressed(
    os.path.join(OUT, "viz_data.npz"),
    phaseA_osi_segs=np.array(phaseA_osi_segs),
    phaseA_osi_means=np.array(phaseA_osi_means),
    phaseA_osi_stds=np.array(phaseA_osi_stds),
    osi_shared=osi_shared,
    pref_shared=pref_shared,
    fine_pres=np.array(fine_pres),
    fine_fwds=np.array(fine_fwds),
    fine_revs=np.array(fine_revs),
    fine_ratios=np.array(fine_ratios),
    fine_wfwds=np.array(fine_wfwds),
    fine_wctrls=np.array(fine_wctrls),
    fine_wpreds=np.array(fine_wpreds),
    ckpt_pres=np.array(ckpt_pres_list),
    ckpt_ratios=np.array(all_ratios),
    ckpt_wpreds=np.array(all_wpreds),
    seq_thetas=np.array(SEQ_THETAS),
)

print(f"\n[seq-figs] All figures saved to: {OUT}")

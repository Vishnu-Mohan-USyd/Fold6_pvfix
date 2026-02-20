#!/usr/bin/env python3
"""debug_pv_contamination.py - Check if test 28 contaminates test 30's W_pv_e.

The self-test runs test 28 (20 plastic presentations with large calibrated W_e_e)
before test 30 (which resets W_e_e to 0.01 but NOT W_pv_e).

This script measures W_pv_e before and after test 28 to quantify contamination,
then tests whether the elevated W_pv_e causes the weight crash.
"""
import os, sys, math
import numpy as np
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from biologically_plausible_v1_stdp import (
    Params, RgcLgnV1Network, compute_osi,
    run_sequence_trial, calibrate_ee_drive,
)

SEED = 42
SEQ_CONTRAST = 2.0

print("=" * 70)
print("DEBUG: Test 28 -> Test 30 W_pv_e contamination check")
print("=" * 70)

# ---- Build network (matching self-test) ----
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

# ---- Phase A: 300 segments ----
print("\n[Phase A] 300 segments...")
net.ff_plastic_enabled = True
net.ee_stdp_active = False
phi = (1.0 + math.sqrt(5.0)) / 2.0
step = 180.0 / phi
offset = float(net.rng.uniform(0.0, 180.0))
for s in range(1, 301):
    th = float((offset + (s - 1) * step) % 180.0)
    net.run_segment(th, plastic=True, contrast=SEQ_CONTRAST)
    if s % 100 == 0:
        ev = np.linspace(0, 180, 12, endpoint=False)
        r = net.evaluate_tuning(ev, repeats=3, contrast=SEQ_CONTRAST)
        o, _ = compute_osi(r, ev)
        print(f"  [seg {s}] OSI={o.mean():.3f}")

ev = np.linspace(0, 180, 12, endpoint=False)
rates = net.evaluate_tuning(ev, repeats=5, contrast=SEQ_CONTRAST)
osi, pref = compute_osi(rates, ev)
print(f"  [Phase A done] OSI={osi.mean():.3f}")

# ---- Snapshot W_pv_e after Phase A ----
wpve_after_phaseA = net.W_pv_e.copy()
mask = net.mask_pv_e
print(f"\n  W_pv_e after Phase A: mean={wpve_after_phaseA[mask].mean():.4f}, "
      f"max={wpve_after_phaseA[mask].max():.4f}")

# ---- Test 28: calibration + 20 plastic presentations ----
print("\n[Test 28] Calibrate E->E + 20 plastic presentations...")
net.ff_plastic_enabled = False
net.ee_stdp_active = True
net._ee_stdp_ramp_factor = 1.0

scale, frac = calibrate_ee_drive(net, 0.15, osi_floor=0.1, contrast=SEQ_CONTRAST)
cal_mean = float(net.W_e_e[net.mask_e_e].mean())
p.w_e_e_max = max(cal_mean * 2.0, p.w_e_e_max)
print(f"  [Cal] scale={scale:.1f}, W_ee mean={cal_mean:.5f}")

wpve_after_cal = net.W_pv_e.copy()
print(f"  W_pv_e after cal: mean={wpve_after_cal[mask].mean():.4f}, "
      f"max={wpve_after_cal[mask].max():.4f}")

# 20 plastic presentations (test 28 Phase B)
t28_seq = [0.0, 45.0, 90.0, 135.0]
net.reset_drive_accumulators()
for _ in range(20):
    run_sequence_trial(net, t28_seq, 30.0, 200.0, SEQ_CONTRAST,
                       plastic=True, vep_mode="spikes")

wpve_after_t28 = net.W_pv_e.copy()
print(f"  W_pv_e after test 28: mean={wpve_after_t28[mask].mean():.4f}, "
      f"max={wpve_after_t28[mask].max():.4f}")
delta = wpve_after_t28 - wpve_after_phaseA
print(f"  W_pv_e change (test28 - phaseA): mean={delta[mask].mean():+.4f}, "
      f"max={delta[mask].max():+.4f}")

# ---- Now simulate test 30 setup ----
# Test 30 resets W_e_e to 0.01 but NOT W_pv_e
print("\n[Test 30 setup] Reset W_e_e=0.01, keep W_pv_e from test 28...")
net.W_e_e[:] = 0.01
np.fill_diagonal(net.W_e_e, 0.0)
net.W_e_e *= net.mask_e_e.astype(np.float32)
p.w_e_e_max = 0.2
net.reset_state()
net.delay_ee_stdp.reset()

# ---- Experiment A: Test 30 with CONTAMINATED W_pv_e ----
print("\n[Exp A] Phase B with contaminated W_pv_e (matching actual self-test)...")
net.ff_plastic_enabled = False
net.ee_stdp_active = True
net._ee_stdp_ramp_factor = 1.0
seq_thetas = [5.0, 65.0, 125.0]
gw = 28.0

for k in range(1, 201):
    res = run_sequence_trial(net, seq_thetas, 30.0, 200.0, SEQ_CONTRAST,
                             plastic=True, vep_mode="spikes")
    if k % 25 == 0:
        wee = float(net.W_e_e[net.mask_e_e].mean())
        wpve = float(net.W_pv_e[mask].mean())
        spk = int(res["v1_counts"].sum())
        # forward/reverse
        fwd_ws, rev_ws = [], []
        for ei in range(len(seq_thetas) - 1):
            d1 = np.minimum(np.abs(pref - seq_thetas[ei]),
                            180.0 - np.abs(pref - seq_thetas[ei]))
            d2 = np.minimum(np.abs(pref - seq_thetas[ei+1]),
                            180.0 - np.abs(pref - seq_thetas[ei+1]))
            for pi in np.where(d2 < gw)[0]:
                for pj in np.where(d1 < gw)[0]:
                    if pi != pj:
                        fwd_ws.append(float(net.W_e_e[pi, pj]))
                        rev_ws.append(float(net.W_e_e[pj, pi]))
        fm = float(np.mean(fwd_ws)) if fwd_ws else 0
        rm = float(np.mean(rev_ws)) if rev_ws else 0
        r = fm / max(1e-10, rm)
        print(f"  [ExpA pres {k:3d}] ratio={r:.3f}  W_ee={wee:.5f}  W_pv_e={wpve:.4f}  spk={spk}")

# ---- Experiment B: Test 30 with CLEAN W_pv_e (restored from Phase A) ----
print("\n[Exp B] Phase B with CLEAN W_pv_e (restored to post-Phase-A value)...")
net.W_e_e[:] = 0.01
np.fill_diagonal(net.W_e_e, 0.0)
net.W_e_e *= net.mask_e_e.astype(np.float32)
p.w_e_e_max = 0.2
net.W_pv_e[:] = wpve_after_phaseA  # <<< CLEAN: restore Phase A W_pv_e
net.reset_state()
net.delay_ee_stdp.reset()

net.ff_plastic_enabled = False
net.ee_stdp_active = True
net._ee_stdp_ramp_factor = 1.0

for k in range(1, 201):
    res = run_sequence_trial(net, seq_thetas, 30.0, 200.0, SEQ_CONTRAST,
                             plastic=True, vep_mode="spikes")
    if k % 25 == 0:
        wee = float(net.W_e_e[net.mask_e_e].mean())
        wpve = float(net.W_pv_e[mask].mean())
        spk = int(res["v1_counts"].sum())
        fwd_ws, rev_ws = [], []
        for ei in range(len(seq_thetas) - 1):
            d1 = np.minimum(np.abs(pref - seq_thetas[ei]),
                            180.0 - np.abs(pref - seq_thetas[ei]))
            d2 = np.minimum(np.abs(pref - seq_thetas[ei+1]),
                            180.0 - np.abs(pref - seq_thetas[ei+1]))
            for pi in np.where(d2 < gw)[0]:
                for pj in np.where(d1 < gw)[0]:
                    if pi != pj:
                        fwd_ws.append(float(net.W_e_e[pi, pj]))
                        rev_ws.append(float(net.W_e_e[pj, pi]))
        fm = float(np.mean(fwd_ws)) if fwd_ws else 0
        rm = float(np.mean(rev_ws)) if rev_ws else 0
        r = fm / max(1e-10, rm)
        print(f"  [ExpB pres {k:3d}] ratio={r:.3f}  W_ee={wee:.5f}  W_pv_e={wpve:.4f}  spk={spk}")

# ---- Experiment C: Test 30 with W_pv_e zeroed ----
print("\n[Exp C] Phase B with W_pv_e ZEROED (like old PV effectively had)...")
net.W_e_e[:] = 0.01
np.fill_diagonal(net.W_e_e, 0.0)
net.W_e_e *= net.mask_e_e.astype(np.float32)
p.w_e_e_max = 0.2
net.W_pv_e[:] = 0.0  # <<< ZERO PV weights
net.reset_state()
net.delay_ee_stdp.reset()

net.ff_plastic_enabled = False
net.ee_stdp_active = True
net._ee_stdp_ramp_factor = 1.0

for k in range(1, 201):
    res = run_sequence_trial(net, seq_thetas, 30.0, 200.0, SEQ_CONTRAST,
                             plastic=True, vep_mode="spikes")
    if k % 25 == 0:
        wee = float(net.W_e_e[net.mask_e_e].mean())
        wpve = float(net.W_pv_e[mask].mean())
        spk = int(res["v1_counts"].sum())
        fwd_ws, rev_ws = [], []
        for ei in range(len(seq_thetas) - 1):
            d1 = np.minimum(np.abs(pref - seq_thetas[ei]),
                            180.0 - np.abs(pref - seq_thetas[ei]))
            d2 = np.minimum(np.abs(pref - seq_thetas[ei+1]),
                            180.0 - np.abs(pref - seq_thetas[ei+1]))
            for pi in np.where(d2 < gw)[0]:
                for pj in np.where(d1 < gw)[0]:
                    if pi != pj:
                        fwd_ws.append(float(net.W_e_e[pi, pj]))
                        rev_ws.append(float(net.W_e_e[pj, pi]))
        fm = float(np.mean(fwd_ws)) if fwd_ws else 0
        rm = float(np.mean(rev_ws)) if rev_ws else 0
        r = fm / max(1e-10, rm)
        print(f"  [ExpC pres {k:3d}] ratio={r:.3f}  W_ee={wee:.5f}  W_pv_e={wpve:.4f}  spk={spk}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Phase A W_pv_e: mean={wpve_after_phaseA[mask].mean():.4f}")
print(f"  After test 28:  mean={wpve_after_t28[mask].mean():.4f}")
print(f"  Contamination:  delta={delta[mask].mean():+.4f}")
print()
print("  If Exp A (contaminated) shows worse learning than Exp B (clean),")
print("  then test 28 is contaminating test 30.")
print("  If Exp C (zeroed PV) shows best learning, then PV inhibition")
print("  itself is the issue, not just contamination.")

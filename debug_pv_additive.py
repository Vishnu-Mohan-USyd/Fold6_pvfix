#!/usr/bin/env python3
"""debug_pv_additive.py - Test additive STDP with old vs new PV params.

The main diagnostic (debug_pv_sequence.py) used weight-dependent STDP and found
PV changes are NOT the cause. But the self-test uses ADDITIVE STDP with W_e_e=0.01.
This script checks if PV changes specifically break the additive STDP path.

Experiments:
  A: New PV (sigma=1.5) + additive STDP (matching self-test)
  B: Old PV (sigma=100) + additive STDP (matching self-test)
  C: New PV + additive STDP + iSTDP OFF
"""

import os
import sys
import math
import time
import numpy as np
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from biologically_plausible_v1_stdp import (
    Params, RgcLgnV1Network, compute_osi,
    run_sequence_trial, calibrate_ee_drive,
)

SEED = 42
PHASE_A_SEGS = 300
SEQ_THETAS = [5.0, 65.0, 125.0]
GROUP_WINDOW = 28.0
SEQ_CONTRAST = 2.0
ELEMENT_MS = 30.0
ITI_MS = 200.0
TOTAL_PRES = 200     # shorter: additive STDP crashes by pres 50 per plan
CKPT_EVERY = 25


def build_network(**overrides) -> Tuple[Params, RgcLgnV1Network]:
    """Build with ADDITIVE STDP matching self-test configuration."""
    kw = dict(
        N=8, M=32, seed=SEED,
        train_segments=0, segment_ms=300,
        train_contrast=SEQ_CONTRAST,
        v1_bias_init=0.0,
        ee_stdp_enabled=True,
        ee_connectivity="all_to_all",
        # Additive STDP (matching self-test)
        ee_stdp_A_plus=0.002,
        ee_stdp_A_minus=0.0024,
        ee_stdp_weight_dep=False,    # ADDITIVE
        rgc_separate_onoff_mosaics=True,
    )
    kw.update(overrides)
    p = Params(**kw)
    net = RgcLgnV1Network(p, init_mode="random")
    return p, net


def run_phase_a(net, p):
    """Phase A: FF STDP for orientation selectivity."""
    net.ff_plastic_enabled = True
    net.ee_stdp_active = False
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    step = 180.0 / phi
    offset = float(net.rng.uniform(0.0, 180.0))
    for s in range(1, PHASE_A_SEGS + 1):
        th = float((offset + (s - 1) * step) % 180.0)
        net.run_segment(th, plastic=True, contrast=SEQ_CONTRAST)
        if s % 100 == 0:
            ev = np.linspace(0, 180, 12, endpoint=False)
            r = net.evaluate_tuning(ev, repeats=3, contrast=SEQ_CONTRAST)
            o, _ = compute_osi(r, ev)
            print(f"    [Phase A seg {s}] OSI={o.mean():.3f}")
    ev = np.linspace(0, 180, 12, endpoint=False)
    rates = net.evaluate_tuning(ev, repeats=5, contrast=SEQ_CONTRAST)
    osi, pref = compute_osi(rates, ev)
    print(f"    [Phase A done] OSI={osi.mean():.3f}")
    return osi, pref


def fwd_rev_ratio(W_e_e, pref):
    """Forward/reverse weight asymmetry ratio."""
    fwd_ws, rev_ws = [], []
    for ei in range(len(SEQ_THETAS) - 1):
        d1 = np.minimum(np.abs(pref - SEQ_THETAS[ei]),
                        180.0 - np.abs(pref - SEQ_THETAS[ei]))
        d2 = np.minimum(np.abs(pref - SEQ_THETAS[ei + 1]),
                        180.0 - np.abs(pref - SEQ_THETAS[ei + 1]))
        pre_idx = np.where(d1 < GROUP_WINDOW)[0]
        post_idx = np.where(d2 < GROUP_WINDOW)[0]
        for pi in post_idx:
            for pj in pre_idx:
                if pi != pj:
                    fwd_ws.append(float(W_e_e[pi, pj]))
                    rev_ws.append(float(W_e_e[pj, pi]))
    if not fwd_ws:
        return 0.0, 0.0, 1.0
    fm = float(np.mean(fwd_ws))
    rm = float(np.mean(rev_ws))
    return fm, rm, fm / max(1e-10, rm)


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


def run_phase_b(net, p, pref, label, total=TOTAL_PRES):
    """Phase B with additive STDP (W_e_e=0.01 fresh start, matching self-test)."""
    # Reset W_e_e to match self-test setup
    net.W_e_e[:] = 0.01
    np.fill_diagonal(net.W_e_e, 0.0)
    net.W_e_e *= net.mask_e_e.astype(np.float32)
    p.w_e_e_max = 0.2
    net.reset_state()
    net.delay_ee_stdp.reset()

    net.ff_plastic_enabled = False
    net.ee_stdp_active = True
    net._ee_stdp_ramp_factor = 1.0

    fm0, rm0, r0 = fwd_rev_ratio(net.W_e_e, pref)
    pres_list = [0]
    ratio_list = [r0]
    w_ee_list = [float(net.W_e_e[net.mask_e_e].mean())]
    w_pve_list = [float(net.W_pv_e[net.mask_pv_e].mean())]
    total_spk_list = []
    fwd_list = [fm0]
    rev_list = [rm0]

    print(f"    [Pre] ratio={r0:.3f}, W_ee={w_ee_list[0]:.5f}, W_pv_e={w_pve_list[0]:.4f}")

    for k in range(1, total + 1):
        res = run_sequence_trial(
            net, SEQ_THETAS, ELEMENT_MS, ITI_MS, SEQ_CONTRAST,
            plastic=True, vep_mode="spikes",
        )

        if k % CKPT_EVERY == 0:
            pres_list.append(k)
            fm, rm, r = fwd_rev_ratio(net.W_e_e, pref)
            fwd_list.append(fm)
            rev_list.append(rm)
            ratio_list.append(r)
            wee = float(net.W_e_e[net.mask_e_e].mean())
            wpve = float(net.W_pv_e[net.mask_pv_e].mean())
            w_ee_list.append(wee)
            w_pve_list.append(wpve)
            tspk = int(res["v1_counts"].sum())
            total_spk_list.append(tspk)

            # Forward/reverse per transition
            parts = []
            for ei in range(len(SEQ_THETAS) - 1):
                d1 = np.minimum(np.abs(pref - SEQ_THETAS[ei]),
                                180.0 - np.abs(pref - SEQ_THETAS[ei]))
                d2 = np.minimum(np.abs(pref - SEQ_THETAS[ei+1]),
                                180.0 - np.abs(pref - SEQ_THETAS[ei+1]))
                pi = np.where(d1 < GROUP_WINDOW)[0]
                po = np.where(d2 < GROUP_WINDOW)[0]
                fw = float(net.W_e_e[np.ix_(po, pi)].mean())
                rv = float(net.W_e_e[np.ix_(pi, po)].mean())
                parts.append(f"{SEQ_THETAS[ei]:.0f}->{SEQ_THETAS[ei+1]:.0f}:{fw:.5f}/{rv:.5f}")

            print(f"    [{label} pres {k:3d}] ratio={r:.3f}  W_ee={wee:.5f}  "
                  f"W_pv_e={wpve:.4f}  spk={tspk}  {', '.join(parts)}")

    return {
        'pres': np.array(pres_list),
        'fwd': np.array(fwd_list),
        'rev': np.array(rev_list),
        'ratio': np.array(ratio_list),
        'w_ee': np.array(w_ee_list),
        'w_pve': np.array(w_pve_list),
        'total_spk': np.array(total_spk_list),
    }


def main():
    t_start = time.time()
    print("=" * 70)
    print("DEBUG: Additive STDP + PV interaction")
    print("  Additive STDP (A+=0.002, A-=0.0024, weight_dep=False)")
    print("  W_e_e = 0.01 (fresh start, matching self-test)")
    print("=" * 70)

    # ---- Exp A: New PV + additive STDP ----
    print("\n[Exp A] New PV (sigma=1.5) + additive STDP")
    p_a, net_a = build_network()
    print("  Phase A:")
    osi_a, pref_a = run_phase_a(net_a, p_a)
    ckpt_a = save_checkpoint(net_a)
    print("  Phase B (additive, W_e_e=0.01):")
    exp_a = run_phase_b(net_a, p_a, pref_a, "ExpA")

    # ---- Exp B: Old PV + additive STDP ----
    print("\n[Exp B] Old PV (sigma=100) + additive STDP")
    p_b, net_b = build_network(pv_in_sigma=100.0, pv_out_sigma=100.0,
                                w_pv_e=0.80, w_pv_e_max=2.0)
    print("  Phase A:")
    osi_b, pref_b = run_phase_a(net_b, p_b)
    print("  Phase B (additive, W_e_e=0.01):")
    exp_b = run_phase_b(net_b, p_b, pref_b, "ExpB")
    del net_b

    # ---- Exp C: New PV + additive STDP + iSTDP OFF ----
    print("\n[Exp C] New PV (sigma=1.5) + additive STDP + iSTDP OFF")
    restore_checkpoint(net_a, ckpt_a)
    p_a.pv_inhib_plastic = False
    print("  Phase B (additive, W_e_e=0.01, iSTDP OFF):")
    exp_c = run_phase_b(net_a, p_a, pref_a, "ExpC")
    p_a.pv_inhib_plastic = True

    # ---- Summary ----
    t_end = time.time()
    print("\n" + "=" * 70)
    print(f"ADDITIVE STDP RESULTS ({t_end - t_start:.0f}s)")
    print("=" * 70)

    ra = float(exp_a['ratio'][-1])
    rb = float(exp_b['ratio'][-1])
    rc = float(exp_c['ratio'][-1])

    print(f"\n  Exp A (new PV, iSTDP ON):  ratio={ra:.3f}  W_ee={exp_a['w_ee'][-1]:.5f}")
    print(f"  Exp B (old PV, iSTDP ON):  ratio={rb:.3f}  W_ee={exp_b['w_ee'][-1]:.5f}")
    print(f"  Exp C (new PV, iSTDP OFF): ratio={rc:.3f}  W_ee={exp_c['w_ee'][-1]:.5f}")

    print(f"\n  W_ee crash check (started at 0.01000):")
    for name, exp in [("A (new PV)", exp_a), ("B (old PV)", exp_b), ("C (no iSTDP)", exp_c)]:
        wees = exp['w_ee']
        crashed = "CRASHED" if wees[-1] < 0.001 else "survived"
        print(f"    {name}: {wees[0]:.5f} -> {wees[-1]:.5f}  ({crashed})")

    print(f"\n  Spike counts (total per trial at checkpoints):")
    for name, exp in [("A (new PV)", exp_a), ("B (old PV)", exp_b), ("C (no iSTDP)", exp_c)]:
        spks = exp['total_spk']
        if len(spks) >= 2:
            print(f"    {name}: first={spks[0]}, last={spks[-1]}")

    # Verdict
    print("\n" + "-" * 50)
    print("VERDICT (Additive STDP):")
    print("-" * 50)
    if rb > ra + 0.1:
        print("  Old PV shows better learning -> PV changes hurt additive STDP")
    elif abs(rb - ra) < 0.1:
        print("  Old and new PV similar -> PV changes NOT the cause for additive STDP either")
    else:
        print("  New PV shows better learning -> PV changes help")

    if rc > ra + 0.1:
        print("  iSTDP OFF helps -> iSTDP is harmful in additive STDP regime")
    elif abs(rc - ra) < 0.1:
        print("  iSTDP has no effect in additive STDP regime")

    all_crashed = all(exp['w_ee'][-1] < 0.001 for exp in [exp_a, exp_b, exp_c])
    if all_crashed:
        print("  ALL experiments crashed W_ee -> additive STDP depression bias is the root cause")
        print("  PV configuration is irrelevant when STDP itself drives all weights to floor")


if __name__ == "__main__":
    main()

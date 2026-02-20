#!/usr/bin/env python3
"""debug_pv_sequence.py - Diagnose why PV inhibition fixes ruined sequence learning.

Hypotheses tested:
  H1: PV iSTDP creates runaway inhibition during Phase B (HIGHEST PRIORITY)
      - iSTDP runs every timestep during Phase B (gated only by p.pv_inhib_plastic)
      - Target rate 8 Hz is below driven rates -> iSTDP potentiates W_pv_e
      - Runaway PV inhibition suppresses E firing -> STDP fails

  H2: Local PV disrupts STDP temporal ordering
      - Local PV creates inhibitory "shadow" that follows sequence

  H3: PV iSTDP leaks through save/restore during collect_traces
      - save_dynamic_state does NOT save W_pv_e
      - NOTE: Code review shows PV iSTDP is inside `if plastic:` block (line 2620),
        so plastic=False during measurement should prevent leaks. Experiment 5 verifies.

Experiments (ordered by diagnostic priority):
  4: Old PV params (sigma=100, w_pv_e=0.80) - regression test (run first)
  1: Disable PV iSTDP during Phase B - tests H1 (primary hypothesis)
  2+3: Track W_pv_e and E firing rates during Phase B - mechanistic detail
  5: W_pv_e save/restore leak quantification - tests H3
"""

import os
import sys
import math
import time
import numpy as np
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from biologically_plausible_v1_stdp import (
    Params, RgcLgnV1Network, compute_osi,
    run_sequence_trial, calibrate_ee_drive,
)

# ======================== Constants ========================
# Match generate_seq_figs.py parameters (weight-dependent STDP, calibrated weights)
SEED = 42
PHASE_A_SEGS = 300
SEQ_THETAS = [5.0, 65.0, 125.0]   # 3 elements, 60 deg apart
GROUP_WINDOW = 28.0
SEQ_CONTRAST = 2.0
ELEMENT_MS = 30.0
ITI_MS = 200.0
TOTAL_PRES = 400       # shorter than 800 for diagnostics; enough to see trends
CKPT_EVERY = 50        # checkpoint frequency

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_pv_output")
os.makedirs(OUT_DIR, exist_ok=True)


# ======================== Helpers ========================

def build_network(**overrides) -> Tuple[Params, RgcLgnV1Network]:
    """Build Params + RgcLgnV1Network with optional param overrides.

    Default: weight-dependent STDP matching generate_seq_figs.py.
    """
    kw = dict(
        N=8, M=32, seed=SEED,
        train_segments=0, segment_ms=300,
        train_contrast=SEQ_CONTRAST,
        v1_bias_init=0.0,
        ee_stdp_enabled=True,
        ee_connectivity="all_to_all",
        ee_stdp_A_plus=0.005,
        ee_stdp_A_minus=0.006,
        ee_stdp_weight_dep=True,
        rgc_separate_onoff_mosaics=True,
    )
    kw.update(overrides)
    p = Params(**kw)
    net = RgcLgnV1Network(p, init_mode="random")
    return p, net


def run_phase_a(net: RgcLgnV1Network, p: Params) -> Tuple[np.ndarray, np.ndarray]:
    """Phase A: FF STDP develops orientation selectivity.

    Returns
    -------
    osi : (M,) array of OSI values per ensemble
    pref : (M,) array of preferred orientations in degrees
    """
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


def do_calibration(net: RgcLgnV1Network, p: Params) -> float:
    """Calibrate E->E drive fraction.

    Returns calibrated W_ee mean.
    """
    net.ff_plastic_enabled = False
    net.ee_stdp_active = True
    net._ee_stdp_ramp_factor = 1.0
    scale, frac = calibrate_ee_drive(net, 0.15, osi_floor=0.1, contrast=SEQ_CONTRAST)
    cal_mean = float(net.W_e_e[net.mask_e_e].mean())
    p.w_e_e_max = max(cal_mean * 2.0, p.w_e_e_max)
    print(f"    [Cal] scale={scale:.1f}, frac={frac:.4f}, W_ee mean={cal_mean:.5f}")
    return cal_mean


def fwd_rev_ratio(W_e_e: np.ndarray, pref: np.ndarray) -> Tuple[float, float, float]:
    """Compute forward/reverse weight asymmetry across all adjacent transitions.

    Returns (forward_mean, reverse_mean, ratio).
    """
    fwd_ws: List[float] = []
    rev_ws: List[float] = []
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


def save_checkpoint(net: RgcLgnV1Network) -> dict:
    """Save all weight matrices + full dynamic state for later restoration."""
    return {
        'W': net.W.copy(),
        'W_e_e': net.W_e_e.copy(),
        'W_pv_e': net.W_pv_e.copy(),
        'dyn': net.save_dynamic_state(),
    }


def restore_checkpoint(net: RgcLgnV1Network, ckpt: dict) -> None:
    """Restore weight matrices + dynamic state from checkpoint."""
    net.W[:] = ckpt['W']
    net.W_e_e[:] = ckpt['W_e_e']
    net.W_pv_e[:] = ckpt['W_pv_e']
    net.restore_dynamic_state(ckpt['dyn'])


def group_indices(pref: np.ndarray) -> Dict[float, np.ndarray]:
    """Map each sequence theta to the indices of neurons preferring that orientation."""
    groups = {}
    for th in SEQ_THETAS:
        d = np.minimum(np.abs(pref - th), 180.0 - np.abs(pref - th))
        groups[th] = np.where(d < GROUP_WINDOW)[0]
    return groups


# ======================== Phase B Runner ========================

def run_phase_b(
    net: RgcLgnV1Network,
    p: Params,
    pref: np.ndarray,
    label: str,
    total: int = TOTAL_PRES,
    track_detail: bool = False,
) -> dict:
    """Run Phase B sequence training with checkpoint measurements.

    Parameters
    ----------
    net : network (state should be at post-calibration checkpoint)
    p : Params (must match the network's params)
    pref : preferred orientations from Phase A evaluation
    label : experiment label for logging
    total : number of sequence presentations
    track_detail : if True, also track W_pv_e range and per-group spike counts

    Returns
    -------
    dict with arrays:
        pres : checkpoint presentation numbers (includes 0 for baseline)
        fwd, rev, ratio : forward/reverse weight stats
        w_ee : mean W_e_e at active connections
        w_pve : mean W_pv_e at active connections
        total_spk : total E spikes per trial at checkpoints (no baseline entry)
        (if track_detail) w_pve_min, w_pve_max : W_pv_e range
        (if track_detail) grp_spk : list of {theta: count} dicts
    """
    net.ff_plastic_enabled = False
    net.ee_stdp_active = True
    net._ee_stdp_ramp_factor = 1.0

    groups = group_indices(pref)

    # Baseline measurements (pres=0, before any training)
    fm0, rm0, r0 = fwd_rev_ratio(net.W_e_e, pref)
    pres_list = [0]
    fwd_list = [fm0]
    rev_list = [rm0]
    ratio_list = [r0]
    w_ee_list = [float(net.W_e_e[net.mask_e_e].mean())]
    w_pve_list = [float(net.W_pv_e[net.mask_pv_e].mean())]
    total_spk_list: List[int] = []  # no baseline; starts at first checkpoint

    if track_detail:
        w_pve_min_list = [float(net.W_pv_e[net.mask_pv_e].min())]
        w_pve_max_list = [float(net.W_pv_e[net.mask_pv_e].max())]
        # Per-group spike counts per trial at checkpoints
        grp_spk_list: List[Dict[float, int]] = []

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

            if track_detail:
                w_pve_min_list.append(float(net.W_pv_e[net.mask_pv_e].min()))
                w_pve_max_list.append(float(net.W_pv_e[net.mask_pv_e].max()))
                gspk = {}
                for th in SEQ_THETAS:
                    idx = groups[th]
                    gspk[th] = int(res["v1_counts"][idx].sum())
                grp_spk_list.append(gspk)

            detail = ""
            if track_detail:
                detail = f"  pve_range=[{w_pve_min_list[-1]:.4f},{w_pve_max_list[-1]:.4f}]  spk={tspk}"
            else:
                detail = f"  spk={tspk}"
            print(f"    [{label} pres {k:3d}] ratio={r:.3f}  W_ee={wee:.5f}  "
                  f"W_pv_e={wpve:.4f}{detail}")

    out: Dict[str, Any] = {
        'pres': np.array(pres_list),
        'fwd': np.array(fwd_list),
        'rev': np.array(rev_list),
        'ratio': np.array(ratio_list),
        'w_ee': np.array(w_ee_list),
        'w_pve': np.array(w_pve_list),
        'total_spk': np.array(total_spk_list),
    }
    if track_detail:
        out['w_pve_min'] = np.array(w_pve_min_list)
        out['w_pve_max'] = np.array(w_pve_max_list)
        out['grp_spk'] = grp_spk_list
    return out


# ======================== Main ========================

def main():
    t_start = time.time()
    print("=" * 70)
    print("DEBUG: Why PV Fixes Ruined Sequence Learning")
    print("=" * 70)
    print(f"  Settings: {TOTAL_PRES} presentations, checkpoint every {CKPT_EVERY}")
    print(f"  Sequence: {SEQ_THETAS} deg, {ELEMENT_MS} ms elements, {ITI_MS} ms ITI")
    print(f"  STDP: weight-dependent (A+={0.005}, A-={0.006})")

    # ================================================================
    # Build & train new-PV network (shared for experiments 1, 2+3, 5)
    # ================================================================
    print("\n[Setup] Building new-PV network (sigma=1.5, w_pv_e=1.0, w_pv_e_max=8.0)...")
    p_new, net_new = build_network()
    print("  Phase A:")
    osi_new, pref_new = run_phase_a(net_new, p_new)
    print("  Calibration:")
    do_calibration(net_new, p_new)
    ckpt_new = save_checkpoint(net_new)
    print(f"  Checkpoint saved. mean OSI={osi_new.mean():.3f}, "
          f"W_pv_e mean={float(net_new.W_pv_e[net_new.mask_pv_e].mean()):.4f}")

    # ================================================================
    # Experiment 4: Old PV params (regression test)
    # Expected: learning works (ratio > 1.3)
    # ================================================================
    print("\n" + "=" * 70)
    print("[Exp 4] Old PV params (sigma=100, w_pv_e=0.80, w_pv_e_max=2.0)")
    print("  Purpose: Confirm this is a PV-change regression, not a pre-existing issue")
    print("  Expected: ratio > 1.3 (learning works with old blanket PV)")
    print("=" * 70)
    p_old, net_old = build_network(
        pv_in_sigma=100.0, pv_out_sigma=100.0,
        w_pv_e=0.80, w_pv_e_max=2.0,
    )
    print("  Phase A:")
    osi_old, pref_old = run_phase_a(net_old, p_old)
    print("  Calibration:")
    do_calibration(net_old, p_old)
    print("  Phase B:")
    exp4 = run_phase_b(net_old, p_old, pref_old, "Exp4")
    del net_old  # free memory
    t_exp4 = time.time()
    print(f"  [Exp 4 done in {t_exp4 - t_start:.0f}s]")

    # ================================================================
    # Experiment 1: Disable PV iSTDP during Phase B
    # Tests H1: PV iSTDP creates runaway inhibition
    # Expected: learning recovers (ratio > 1.3)
    # ================================================================
    print("\n" + "=" * 70)
    print("[Exp 1] Disable PV iSTDP during Phase B (pv_inhib_plastic=False)")
    print("  Purpose: Test H1 - is PV iSTDP the primary cause?")
    print("  Expected: ratio > 1.3 if H1 is correct")
    print("=" * 70)
    restore_checkpoint(net_new, ckpt_new)
    p_new.pv_inhib_plastic = False   # <<< KEY: disable iSTDP
    print("  Phase B (pv_inhib_plastic=False):")
    exp1 = run_phase_b(net_new, p_new, pref_new, "Exp1")
    p_new.pv_inhib_plastic = True    # restore for next experiment
    t_exp1 = time.time()
    print(f"  [Exp 1 done in {t_exp1 - t_exp4:.0f}s]")

    # ================================================================
    # Experiments 2+3: Baseline with iSTDP + detailed tracking
    # Tracks W_pv_e evolution and E firing rates
    # Expected: W_pv_e increases monotonically, E rates decrease
    # ================================================================
    print("\n" + "=" * 70)
    print("[Exp 2+3] Baseline with PV iSTDP ON + detailed tracking")
    print("  Purpose: Characterize the failure mechanism")
    print("  Expected: W_pv_e increases, E spike counts decrease over training")
    print("=" * 70)
    restore_checkpoint(net_new, ckpt_new)
    print("  Phase B (full tracking):")
    exp23 = run_phase_b(net_new, p_new, pref_new, "Exp23", track_detail=True)
    t_exp23 = time.time()
    print(f"  [Exp 2+3 done in {t_exp23 - t_exp1:.0f}s]")

    # ================================================================
    # Experiment 5: W_pv_e save/restore leak test
    # Tests H3: Does W_pv_e drift during non-plastic measurement trials?
    # NOTE: Code review shows PV iSTDP is inside `if plastic:` (line 2620),
    #       so plastic=False should prevent any W_pv_e modification.
    # ================================================================
    print("\n" + "=" * 70)
    print("[Exp 5] W_pv_e save/restore leak quantification")
    print("  Purpose: Test H3 - does W_pv_e drift during collect_traces?")
    print("  Note: PV iSTDP is inside `if plastic:` block, so expect NO leak")
    print("=" * 70)
    restore_checkpoint(net_new, ckpt_new)
    net_new.ff_plastic_enabled = False
    net_new.ee_stdp_active = False

    wpve_before = net_new.W_pv_e.copy()

    # Mimic what collect_traces does: save_dynamic_state, run non-plastic trials, restore
    snap5 = net_new.save_dynamic_state()
    for i in range(15):
        net_new.reset_state()
        run_sequence_trial(
            net_new, SEQ_THETAS, ELEMENT_MS, ITI_MS, SEQ_CONTRAST,
            plastic=False,
        )
    net_new.restore_dynamic_state(snap5)

    wpve_after = net_new.W_pv_e.copy()
    abs_drift = float(np.abs(wpve_after - wpve_before).max())
    ref_scale = float(np.abs(wpve_before).max())
    rel_drift = abs_drift / max(1e-10, ref_scale)
    print(f"  W_pv_e max absolute drift after 15 non-plastic trials: {abs_drift:.2e}")
    print(f"  W_pv_e reference max value: {ref_scale:.4f}")
    print(f"  W_pv_e max relative drift: {rel_drift:.2e}")
    h3_confirmed = abs_drift > 1e-6
    print(f"  H3 (save/restore leak): {'CONFIRMED - W_pv_e changed!' if h3_confirmed else 'REJECTED - no leak (as expected from code review)'}")

    # ================================================================
    # Summary
    # ================================================================
    t_end = time.time()
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print(f"  Total runtime: {t_end - t_start:.0f}s")
    print("=" * 70)

    r4_final = float(exp4['ratio'][-1])
    r1_final = float(exp1['ratio'][-1])
    r23_final = float(exp23['ratio'][-1])

    print(f"\n  Exp 4 (old PV, sigma=100):     final ratio = {r4_final:.3f}  "
          f"{'PASS (learning works)' if r4_final > 1.3 else 'FAIL (no learning)'}")
    print(f"  Exp 1 (new PV, iSTDP OFF):     final ratio = {r1_final:.3f}  "
          f"{'RECOVERED' if r1_final > 1.3 else 'still broken'}")
    print(f"  Exp 2+3 (new PV, iSTDP ON):    final ratio = {r23_final:.3f}  "
          f"(baseline, expected broken)")

    # W_pv_e evolution from Exp 2+3
    print(f"\n  W_pv_e evolution (Exp 2+3, iSTDP ON):")
    print(f"    start:  mean={exp23['w_pve'][0]:.4f}")
    print(f"    end:    mean={exp23['w_pve'][-1]:.4f}")
    if 'w_pve_min' in exp23:
        print(f"    end range: [{exp23['w_pve_min'][-1]:.4f}, {exp23['w_pve_max'][-1]:.4f}]")
    pve_delta = exp23['w_pve'][-1] - exp23['w_pve'][0]
    pve_direction = 'INCREASED' if pve_delta > 0 else 'DECREASED'
    print(f"    change: {pve_delta:+.4f} ({pve_direction})")

    # W_pv_e evolution from Exp 1 (iSTDP OFF - should be static)
    print(f"\n  W_pv_e with iSTDP OFF (Exp 1):")
    print(f"    start:  mean={exp1['w_pve'][0]:.4f}")
    print(f"    end:    mean={exp1['w_pve'][-1]:.4f}")
    print(f"    change: {exp1['w_pve'][-1] - exp1['w_pve'][0]:+.4f} (should be 0.0000)")

    # W_ee evolution comparison
    print(f"\n  W_ee evolution comparison:")
    print(f"    Exp 4 (old PV):       {exp4['w_ee'][0]:.5f} -> {exp4['w_ee'][-1]:.5f}")
    print(f"    Exp 1 (iSTDP OFF):    {exp1['w_ee'][0]:.5f} -> {exp1['w_ee'][-1]:.5f}")
    print(f"    Exp 2+3 (iSTDP ON):   {exp23['w_ee'][0]:.5f} -> {exp23['w_ee'][-1]:.5f}")

    # Spike count evolution (Exp 2+3 vs Exp 1)
    if len(exp23['total_spk']) >= 2:
        print(f"\n  E total spikes per trial:")
        print(f"    Exp 1 (iSTDP OFF):  first={exp1['total_spk'][0] if len(exp1['total_spk']) > 0 else 'N/A'}, "
              f"last={exp1['total_spk'][-1] if len(exp1['total_spk']) > 0 else 'N/A'}")
        print(f"    Exp 2+3 (iSTDP ON): first={exp23['total_spk'][0]}, last={exp23['total_spk'][-1]}")

    # Per-group spike counts (Exp 2+3)
    if 'grp_spk' in exp23 and len(exp23['grp_spk']) >= 2:
        print(f"\n  Per-group spike counts over training (Exp 2+3):")
        first_grp = exp23['grp_spk'][0]
        last_grp = exp23['grp_spk'][-1]
        for th in SEQ_THETAS:
            print(f"    {th:5.0f} deg: {first_grp[th]:3d} -> {last_grp[th]:3d}  "
                  f"({'suppressed' if last_grp[th] < first_grp[th] * 0.7 else 'stable'})")

    print(f"\n  H3 save/restore leak: {'CONFIRMED' if h3_confirmed else 'REJECTED'} "
          f"(drift={abs_drift:.2e})")

    # ---- Hypothesis verdicts ----
    print("\n" + "-" * 50)
    print("HYPOTHESIS VERDICTS:")
    print("-" * 50)

    # Is this a PV regression?
    if r4_final > 1.3 and r23_final < 1.2:
        print("  [REGRESSION] PV parameter change IS the regression source")
        print("    Evidence: old params (Exp 4) pass, new params (Exp 2+3) fail")
    elif r4_final < 1.3:
        print(f"  [WARNING] Old PV params also fail (ratio={r4_final:.3f})")
        print("    The regression may pre-date PV changes, or 400 pres is too few")
    else:
        print("  [INCONCLUSIVE] Both old and new PV params show similar learning")

    # H1: PV iSTDP runaway
    if r1_final > 1.3 and r23_final < 1.2:
        print("\n  [H1 CONFIRMED] PV iSTDP is the primary cause")
        print("    Evidence: disabling iSTDP (Exp 1) recovered learning")
        if pve_direction == 'INCREASED':
            print("    Mechanism: iSTDP increases W_pv_e -> runaway PV inhibition")
            print("               -> E suppression -> insufficient STDP correlations")
        else:
            print("    Mechanism: iSTDP modifies W_pv_e dynamics -> disrupts temporal ordering")
        print("    Recommendation: Freeze iSTDP during Phase B, or slow its learning rate,")
        print("    or raise target_rate_hz to match driven rate during sequences")
    elif r1_final < 1.3 and r23_final < 1.2:
        print("\n  [H1 NOT CONFIRMED] Disabling iSTDP did not recover learning")
        print("    Consider H2 (local PV temporal disruption) as alternative")
    else:
        print("\n  [H1 INCONCLUSIVE] Both conditions show learning (or both fail)")

    # H3: Save/restore leak
    if h3_confirmed:
        print(f"\n  [H3 CONFIRMED] W_pv_e leaks during non-plastic trials (drift={abs_drift:.2e})")
        print("    Bug: W_pv_e is modified even when plastic=False")
    else:
        print(f"\n  [H3 REJECTED] No W_pv_e leak during non-plastic trials")
        print("    PV iSTDP is correctly gated by `if plastic:` block")

    # ---- Save detailed data ----
    save_path = os.path.join(OUT_DIR, "debug_results.npz")
    save_dict: Dict[str, Any] = {}
    for name, exp in [('exp4', exp4), ('exp1', exp1), ('exp23', exp23)]:
        for key in ['pres', 'fwd', 'rev', 'ratio', 'w_ee', 'w_pve', 'total_spk']:
            if key in exp:
                save_dict[f'{name}_{key}'] = exp[key]
        if 'w_pve_min' in exp:
            save_dict[f'{name}_w_pve_min'] = exp['w_pve_min']
            save_dict[f'{name}_w_pve_max'] = exp['w_pve_max']
    save_dict['exp5_abs_drift'] = np.float64(abs_drift)
    save_dict['exp5_rel_drift'] = np.float64(rel_drift)
    np.savez_compressed(save_path, **save_dict)
    print(f"\nDetailed data saved to {save_path}")
    print(f"Total runtime: {time.time() - t_start:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()

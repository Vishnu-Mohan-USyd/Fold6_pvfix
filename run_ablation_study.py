#!/usr/bin/env python3
"""Systematic component ablation study for the RGC-LGN-V1 network.

Tests whether each component (PV, SOM, STP, heterosynaptic,
ON/OFF split, triplet STDP, retinotopy, E→E connectivity, E→E plasticity)
actually contributes to network function (OSI emergence, sequence learning)
vs. being "namesake" — via systematic, controlled ablation experiments.

Protocol matches validate_sequence.py:
  - Additive STDP: ee_stdp_weight_dep=False, A+=0.002, A-=0.0024
  - M=32, N=8
  - Phase A: 300 segments of golden-ratio feedforward STDP
  - Calibrate E→E drive to 15%
  - Phase B: 600 sequence presentations (ABCD = [0, 45, 90, 135]°)

References:
  - Gavornik & Bear (2014) "Learned spatiotemporal sequence recognition
    and prediction in primary visual cortex"
  - Izhikevich (2003, 2007), Pfister & Gerstner (2006)

Usage:
  python run_ablation_study.py                    # full study (all conditions, 5 seeds)
  python run_ablation_study.py --keys BL A5       # run specific conditions
  python run_ablation_study.py --seeds 1 2        # run specific seeds
  python run_ablation_study.py --quick             # single seed, BL + A5 only (verify pipeline)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from biologically_plausible_v1_stdp import (
    Params,
    RgcLgnV1Network,
    compute_osi,
    run_sequence_trial,
    calibrate_ee_drive,
    circ_mean_resultant_180,
    max_circ_gap_180,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants (matching validate_sequence.py)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
M_DEFAULT = 32
N_DEFAULT = 8
SEQ_THETAS = [0.0, 45.0, 90.0, 135.0]
ELEMENT_MS = 30.0
ITI_MS = 200.0
CONTRAST = 2.0
PHASE_A_SEGMENTS = 300
PHASE_B_PRESENTATIONS = 600
TARGET_EE_DRIVE = 0.15
TEST_REPEATS = 30
OUT_DIR = "ablation_results"

# Base Params: additive STDP (matching validate_sequence.py)
BASE_PARAMS: Dict[str, Any] = dict(
    N=N_DEFAULT,
    M=M_DEFAULT,
    train_segments=0,
    train_stimulus="grating",
    train_contrast=CONTRAST,
    ee_stdp_enabled=True,
    ee_connectivity="all_to_all",
    ee_stdp_A_plus=0.002,
    ee_stdp_A_minus=0.0024,
    ee_stdp_weight_dep=False,
)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment definitions
# ──────────────────────────────────────────────────────────────────────────────
def _hook_b9(net: RgcLgnV1Network) -> None:
    """E→E OFF: zero all E→E weights."""
    net.W_e_e[:] = 0.0


def _hook_b11(net: RgcLgnV1Network) -> None:
    """SOM OFF in Phase B: zero SOM→E weights."""
    net.W_som_e[:] = 0.0


def _hook_b12(net: RgcLgnV1Network) -> None:
    """PV OFF in Phase B: zero PV→E weights."""
    net.W_pv_e[:] = 0.0


# Each experiment: {
#   "name": display name,
#   "category": "A" | "B" | "C",
#   "param_overrides": dict merged into BASE_PARAMS at construction,
#   "phase_b_hook": callable(net) -> None, applied after Phase A + calibration,
#   "ee_stdp_in_phase_b": bool (default True),
#   "skip_ee_calibration": bool (default False),
#   "hypothesis": str,
#   "bio_description": str,
# }
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "BL": {
        "name": "Full model (baseline)",
        "category": "A",
        "param_overrides": {},
        "phase_b_hook": None,
        "hypothesis": "Reference baseline — all components active",
        "bio_description": (
            "Complete RGC→LGN→V1 network with PV/SOM inhibition, "
            "thalamocortical STP, triplet STDP with heterosynaptic depression, "
            "ON/OFF split constraint, and retinotopic locality."
        ),
    },
    "A1": {
        "name": "PV inhibition OFF",
        "category": "A",
        "param_overrides": {"w_pv_e": 0.0, "pv_inhib_plastic": False},
        "phase_b_hook": None,
        "hypothesis": (
            "PV provides fast feedforward gain control (sharpens OS via "
            "divisive normalization). Without PV, OSI drops and rates increase."
        ),
        "bio_description": (
            "PV (fast-spiking) interneurons are disabled. PV→E inhibitory "
            "conductance set to zero and PV iSTDP plasticity disabled."
        ),
    },
    "A2": {
        "name": "SOM inhibition OFF",
        "category": "A",
        "param_overrides": {"w_som_e": 0.0},
        "phase_b_hook": None,
        "hypothesis": (
            "SOM provides surround suppression and diversifies orientation "
            "preference across ensembles. Without SOM, preferences clump."
        ),
        "bio_description": (
            "SOM (low-threshold spiking) interneurons' output is silenced. "
            "SOM→E lateral inhibitory conductance set to zero."
        ),
    },
    "A4": {
        "name": "STP OFF",
        "category": "A",
        "param_overrides": {"tc_stp_enabled": False, "tc_stp_pv_enabled": False},
        "phase_b_hook": None,
        "hypothesis": (
            "Thalamocortical short-term depression provides fast local gain "
            "control. Without STP, rates may destabilize."
        ),
        "bio_description": (
            "Short-term synaptic depression at LGN→E and LGN→PV synapses "
            "disabled (tc_stp_enabled=False, tc_stp_pv_enabled=False)."
        ),
    },
    "A5": {
        "name": "Heterosynaptic OFF",
        "category": "A",
        "param_overrides": {"A_het": 0.0},
        "phase_b_hook": None,
        "hypothesis": (
            "Heterosynaptic depression is the primary weight competition "
            "mechanism. Without it, weights saturate and OSI collapses."
        ),
        "bio_description": (
            "Heterosynaptic (resource-like) depression disabled (A_het=0). "
            "Postsynaptic spikes no longer weaken inactive synapses."
        ),
    },
    "A6": {
        "name": "ON/OFF split OFF",
        "category": "A",
        "param_overrides": {"A_split": 0.0, "split_constraint_rate": 0.0},
        "phase_b_hook": None,
        "hypothesis": (
            "ON/OFF split promotes development of phase-opponent subfields. "
            "Without it, RF structure may degrade under grating stimuli."
        ),
        "bio_description": (
            "ON/OFF developmental split constraint disabled (A_split=0, "
            "split_constraint_rate=0). ON and OFF channels unconstrained."
        ),
    },
    "A7": {
        "name": "Triplet→Pair STDP",
        "category": "A",
        "param_overrides": {"A3_plus": 0.0, "A3_minus": 0.0},
        "phase_b_hook": None,
        "hypothesis": (
            "Triplet terms aid convergence speed and robustness. With only "
            "pair STDP, learning may be slower but still functional."
        ),
        "bio_description": (
            "Triplet STDP enhancement disabled (A3_plus=0, A3_minus=0). "
            "Reverts to standard pair-based STDP (Bi & Poo 1998)."
        ),
    },
    "A8": {
        "name": "Retinotopy OFF",
        "category": "A",
        "param_overrides": {
            "lgn_sigma_e": 100.0,
            "lgn_sigma_pv": 100.0,
        },
        "phase_b_hook": None,
        "hypothesis": (
            "Retinotopic locality is essential for spatial structure. "
            "Without it, OSI drops to ~0 (all inputs equally weighted)."
        ),
        "bio_description": (
            "Retinotopic locality envelopes set to sigma=100 pixels "
            "(effectively uniform/global). No spatial structure constraint."
        ),
    },
    # ── Category B: Ablation during Phase B only ──
    "B9": {
        "name": "E→E OFF (Phase B)",
        "category": "B",
        "param_overrides": {},
        "phase_b_hook": _hook_b9,
        "ee_stdp_in_phase_b": False,
        "skip_ee_calibration": True,
        "hypothesis": (
            "No E→E substrate at all. Without recurrent connections, "
            "there is no mechanism for sequence learning."
        ),
        "bio_description": (
            "All E→E lateral weights zeroed before Phase B. "
            "E→E STDP disabled. Tests necessity of recurrent substrate."
        ),
    },
    "B10": {
        "name": "E→E STDP OFF (Phase B)",
        "category": "B",
        "param_overrides": {},
        "phase_b_hook": None,
        "ee_stdp_in_phase_b": False,
        "hypothesis": (
            "Calibrated E→E weights are present but fixed. Without "
            "plasticity, no forward/reverse asymmetry should develop."
        ),
        "bio_description": (
            "E→E weights remain at calibrated values but STDP is disabled "
            "during Phase B. Tests necessity of recurrent plasticity."
        ),
    },
    "B11": {
        "name": "SOM OFF (Phase B only)",
        "category": "B",
        "param_overrides": {},
        "phase_b_hook": _hook_b11,
        "hypothesis": (
            "SOM maintains sequence specificity by suppressing non-target "
            "ensembles. Without SOM, potentiation is less selective."
        ),
        "bio_description": (
            "SOM→E weights zeroed before Phase B (SOM was active during "
            "Phase A for normal OSI development)."
        ),
    },
    "B12": {
        "name": "PV OFF (Phase B only)",
        "category": "B",
        "param_overrides": {},
        "phase_b_hook": _hook_b12,
        "hypothesis": (
            "PV controls firing rates for proper STDP timing windows. "
            "Without PV during learning, STDP becomes noisier."
        ),
        "bio_description": (
            "PV→E weights zeroed before Phase B (PV was active during "
            "Phase A for normal OSI development)."
        ),
    },
    # ── Category C: Interaction tests ──
    "C13": {
        "name": "PV + SOM both OFF",
        "category": "C",
        "param_overrides": {
            "w_pv_e": 0.0,
            "w_som_e": 0.0,
            "pv_inhib_plastic": False,
        },
        "phase_b_hook": None,
        "hypothesis": (
            "All cortical inhibition removed. Expect catastrophic failure: "
            "runaway excitation, no orientation selectivity."
        ),
        "bio_description": (
            "Both PV and SOM inhibitory pathways disabled. No cortical "
            "inhibition remains."
        ),
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Metrics collection
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_phase_a_metrics(net: RgcLgnV1Network) -> Dict[str, Any]:
    """Collect Phase A metrics: OSI, preference distribution, rates, ON/OFF correlation."""
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = net.evaluate_tuning(thetas_eval, repeats=5, contrast=CONTRAST)
    osi, pref = compute_osi(rates, thetas_eval)

    osi_mean = float(osi.mean())
    osi_median = float(np.median(osi))
    n_tuned = int((osi >= 0.3).sum())

    # Preference distribution of tuned neurons
    tuned_mask = osi >= 0.3
    if tuned_mask.sum() > 1:
        pref_tuned = pref[tuned_mask]
        R, _ = circ_mean_resultant_180(pref_tuned)
        pref_uniformity_R = float(R)
        pref_max_gap_deg = float(max_circ_gap_180(pref_tuned))
    else:
        pref_uniformity_R = 0.0
        pref_max_gap_deg = 180.0

    mean_rate_hz = float(rates.mean())

    # ON/OFF weight correlation per neuron
    n_pix = net.p.N * net.p.N
    W_on = net.W[:, :n_pix]
    W_off = net.W[:, n_pix:]
    onoff_corrs = []
    for i in range(net.M):
        if W_on[i].std() > 1e-8 and W_off[i].std() > 1e-8:
            c = float(np.corrcoef(W_on[i], W_off[i])[0, 1])
            onoff_corrs.append(c if not math.isnan(c) else 0.0)
        else:
            onoff_corrs.append(0.0)
    onoff_corr = float(np.mean(onoff_corrs))

    return {
        "osi_mean": osi_mean,
        "osi_median": osi_median,
        "n_tuned": n_tuned,
        "pref_uniformity_R": pref_uniformity_R,
        "pref_max_gap": pref_max_gap_deg,
        "mean_rate_hz": mean_rate_hz,
        "onoff_corr": onoff_corr,
        # Raw arrays for downstream use (not serialized to JSON)
        "_osi": osi,
        "_pref": pref,
        "_rates": rates,
    }


def evaluate_stability(net: RgcLgnV1Network) -> Dict[str, float]:
    """Quick stability check: run one segment and report cell-type firing rates."""
    counts = net.run_segment_counts(90.0, plastic=False, contrast=CONTRAST)
    duration_s = net.p.segment_ms / 1000.0

    v1_rates = counts["v1_counts"].astype(np.float64) / duration_s
    pv_rates = counts["pv_counts"].astype(np.float64) / duration_s
    som_rates = counts["som_counts"].astype(np.float64) / duration_s

    return {
        "v1_rate_hz": float(v1_rates.mean()),
        "pv_rate_hz": float(pv_rates.mean()),
        "som_rate_hz": float(som_rates.mean()),
        "n_silent_e": int((v1_rates < 0.5).sum()),
    }


def evaluate_phase_b_metrics(
    net: RgcLgnV1Network,
    pref_deg: np.ndarray,
) -> Dict[str, Any]:
    """Collect Phase B metrics: weight asymmetry, spike selectivity, omission signal, OSI."""
    # ── Neuron grouping by preferred orientation ──
    groups: Dict[float, np.ndarray] = {}
    for th in SEQ_THETAS:
        diffs = np.abs(pref_deg - th)
        diffs = np.minimum(diffs, 180.0 - diffs)
        groups[th] = np.where(diffs < 22.5)[0]

    # ── Forward / reverse weight analysis ──
    fwd_weights: List[float] = []
    rev_weights: List[float] = []
    for i in range(len(SEQ_THETAS) - 1):
        pre_group = groups[SEQ_THETAS[i]]
        post_group = groups[SEQ_THETAS[i + 1]]
        for post_idx in post_group:
            for pre_idx in pre_group:
                if post_idx != pre_idx:
                    fwd_weights.append(float(net.W_e_e[post_idx, pre_idx]))
                    rev_weights.append(float(net.W_e_e[pre_idx, post_idx]))

    fwd_mean = float(np.mean(fwd_weights)) if fwd_weights else 0.0
    rev_mean = float(np.mean(rev_weights)) if rev_weights else 0.0
    fwd_rev_weight_ratio = fwd_mean / max(1e-10, rev_mean)

    # ── Trained vs reverse spike counts ──
    trained_counts: List[float] = []
    reverse_counts: List[float] = []
    for rep in range(TEST_REPEATS):
        net.reset_state()
        r_t = run_sequence_trial(
            net, SEQ_THETAS, ELEMENT_MS, ITI_MS,
            CONTRAST, plastic=False, record=False, vep_mode="spikes",
        )
        trained_counts.append(float(sum(c.sum() for c in r_t["element_counts"])))

        net.reset_state()
        r_r = run_sequence_trial(
            net, SEQ_THETAS[::-1], ELEMENT_MS, ITI_MS,
            CONTRAST, plastic=False, record=False, vep_mode="spikes",
        )
        reverse_counts.append(float(sum(c.sum() for c in r_r["element_counts"])))

    trained_spike_mean = float(np.mean(trained_counts))
    reverse_spike_mean = float(np.mean(reverse_counts))
    spike_ratio = trained_spike_mean / max(1e-10, reverse_spike_mean)

    # ── Omission signal ──
    omit_signals: List[float] = []
    for rep in range(TEST_REPEATS):
        net.reset_state()
        r_omit = run_sequence_trial(
            net, SEQ_THETAS, ELEMENT_MS, ITI_MS,
            CONTRAST, plastic=False, record=True,
            omit_index=2, vep_mode="i_exc",
        )
        omit_signals.append(float(np.mean(r_omit["element_traces"][2])))
    omission_signal = float(np.mean(omit_signals))

    # ── OSI preservation ──
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates_post = net.evaluate_tuning(thetas_eval, repeats=5, contrast=CONTRAST)
    osi_post, _ = compute_osi(rates_post, thetas_eval)

    # ── Stability after Phase B ──
    stability = evaluate_stability(net)

    return {
        "fwd_rev_weight_ratio": fwd_rev_weight_ratio,
        "fwd_weight_mean": fwd_mean,
        "rev_weight_mean": rev_mean,
        "trained_spike_mean": trained_spike_mean,
        "reverse_spike_mean": reverse_spike_mean,
        "spike_ratio": spike_ratio,
        "omission_signal": omission_signal,
        "osi_post": float(osi_post.mean()),
        **{f"stability_{k}": v for k, v in stability.items()},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core training loop
# ──────────────────────────────────────────────────────────────────────────────
def run_phase_a(net: RgcLgnV1Network, n_segments: int) -> None:
    """Phase A: feedforward STDP with golden-ratio orientation sweep."""
    net.ff_plastic_enabled = True
    net.ee_stdp_active = False

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    theta_step = 180.0 / phi
    theta_offset = float(net.rng.uniform(0.0, 180.0))

    for s in range(1, n_segments + 1):
        th = float((theta_offset + (s - 1) * theta_step) % 180.0)
        net.run_segment(th, plastic=True, contrast=CONTRAST)

        if s % max(1, n_segments // 5) == 0 or s == n_segments:
            thetas_eval = np.linspace(0, 180, 12, endpoint=False)
            rates = net.evaluate_tuning(thetas_eval, repeats=3, contrast=CONTRAST)
            osi, _ = compute_osi(rates, thetas_eval)
            print(f"    [Phase A seg {s}/{n_segments}] OSI={osi.mean():.3f}, "
                  f"rate={rates.mean():.2f} Hz")


def run_phase_b(
    net: RgcLgnV1Network,
    n_presentations: int,
    ee_stdp_active: bool = True,
) -> None:
    """Phase B: sequence training with optional E→E STDP."""
    net.ff_plastic_enabled = False
    net.ee_stdp_active = ee_stdp_active
    net._ee_stdp_ramp_factor = 1.0

    log_interval = max(1, n_presentations // 10)
    for k in range(1, n_presentations + 1):
        run_sequence_trial(
            net, SEQ_THETAS, ELEMENT_MS, ITI_MS,
            CONTRAST, plastic=True, record=False, vep_mode="spikes",
        )

        if k % log_interval == 0:
            drive_frac, _ = net.get_drive_fraction()
            net.reset_drive_accumulators()
            off_diag = net.W_e_e[net.mask_e_e]
            print(f"    [Phase B pres {k}/{n_presentations}] "
                  f"drive_frac={drive_frac:.4f}, "
                  f"W_ee mean={off_diag.mean():.5f} max={off_diag.max():.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Single experiment runner
# ──────────────────────────────────────────────────────────────────────────────
def run_single_experiment(
    key: str,
    seed: int,
    *,
    baseline_checkpoint: Optional[Tuple[RgcLgnV1Network, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run one experiment for one seed. Returns a flat metrics dict.

    Parameters
    ----------
    key : str
        Experiment key (e.g. "BL", "A5", "B10").
    seed : int
        Random seed.
    baseline_checkpoint : tuple or None
        For Category B experiments: (deep-copied network after Phase A + calibration,
        phase_a_metrics dict). Avoids re-running Phase A.

    Returns
    -------
    metrics : dict
        Combined Phase A + Phase B metrics, plus metadata.
    """
    exp = EXPERIMENTS[key]
    category = exp["category"]
    param_overrides = exp.get("param_overrides", {})
    phase_b_hook = exp.get("phase_b_hook", None)
    ee_stdp_in_phase_b = exp.get("ee_stdp_in_phase_b", True)
    skip_ee_calibration = exp.get("skip_ee_calibration", False)

    t_start = time.time()
    print(f"\n{'='*70}")
    print(f"  Experiment: {key} — {exp['name']}  (seed={seed})")
    print(f"{'='*70}")

    metrics: Dict[str, Any] = {
        "key": key,
        "seed": seed,
        "name": exp["name"],
        "category": category,
    }

    # ── Build or restore network ──
    # Use checkpoint when available (run_all only passes it for BL + Category B)
    if baseline_checkpoint is not None:
        # Restore from Phase A checkpoint
        net = copy.deepcopy(baseline_checkpoint[0])
        phase_a_metrics = dict(baseline_checkpoint[1])  # copy metadata
        print(f"  [Restored baseline Phase A checkpoint for Category B]")
    else:
        # Build fresh network with overrides
        p_kwargs = dict(BASE_PARAMS)
        p_kwargs["seed"] = seed
        p_kwargs.update(param_overrides)
        p = Params(**p_kwargs)
        net = RgcLgnV1Network(p, init_mode="random")

        # ── Phase A ──
        print(f"  [Phase A] {PHASE_A_SEGMENTS} segments...")
        run_phase_a(net, PHASE_A_SEGMENTS)

        # ── Evaluate Phase A ──
        phase_a_metrics = evaluate_phase_a_metrics(net)
        stability_a = evaluate_stability(net)
        phase_a_metrics.update({f"stability_a_{k}": v for k, v in stability_a.items()})
        print(f"  [Phase A] OSI={phase_a_metrics['osi_mean']:.3f}, "
              f"n_tuned={phase_a_metrics['n_tuned']}/{net.M}, "
              f"rate={phase_a_metrics['mean_rate_hz']:.2f} Hz")

    # Store Phase A metrics (exclude numpy arrays)
    for k_m, v_m in phase_a_metrics.items():
        if not k_m.startswith("_"):
            metrics[f"phase_a_{k_m}"] = v_m

    # Retrieve pref_deg for neuron grouping
    pref_deg = phase_a_metrics.get("_pref", None)
    if pref_deg is None:
        # Re-compute if not available (shouldn't happen)
        thetas_eval = np.linspace(0, 180, 12, endpoint=False)
        rates = net.evaluate_tuning(thetas_eval, repeats=5, contrast=CONTRAST)
        _, pref_deg = compute_osi(rates, thetas_eval)

    # ── Apply Phase B hook (Category B) ──
    if phase_b_hook is not None:
        print(f"  [Phase B hook] Applying: {phase_b_hook.__doc__}")
        phase_b_hook(net)

    # ── Calibrate E→E drive ──
    # Skip if: (a) explicitly disabled, or (b) checkpoint already includes calibration
    checkpoint_already_calibrated = baseline_checkpoint is not None
    if skip_ee_calibration:
        print(f"  [Calibrate] Skipped (E→E disabled for this experiment)")
    elif checkpoint_already_calibrated:
        # Checkpoint from _build_base_checkpoint already includes calibration
        print(f"  [Calibrate] Skipped (using pre-calibrated checkpoint)")
        # Copy calibration metadata from checkpoint
        for ck in ("ee_calibration_scale", "ee_calibration_frac", "ee_calibration_mean"):
            if ck in phase_a_metrics:
                metrics[ck] = phase_a_metrics[ck]
    else:
        print(f"  [Calibrate] Target E→E drive = {TARGET_EE_DRIVE:.2f}...")
        try:
            scale, frac = calibrate_ee_drive(net, TARGET_EE_DRIVE, contrast=CONTRAST)
            off_diag = net.W_e_e[net.mask_e_e]
            cal_mean = float(off_diag.mean())
            print(f"  [Calibrate] scale={scale:.1f}, frac={frac:.4f}, "
                  f"W_ee mean={cal_mean:.5f}")
            metrics["ee_calibration_scale"] = float(scale)
            metrics["ee_calibration_frac"] = float(frac)
            metrics["ee_calibration_mean"] = cal_mean
        except Exception as e:
            print(f"  [Calibrate] FAILED: {e}")
            metrics["ee_calibration_error"] = str(e)

    # ── Phase B ──
    print(f"  [Phase B] {PHASE_B_PRESENTATIONS} presentations "
          f"(ee_stdp={'ON' if ee_stdp_in_phase_b else 'OFF'})...")
    run_phase_b(net, PHASE_B_PRESENTATIONS, ee_stdp_active=ee_stdp_in_phase_b)

    # ── Evaluate Phase B ──
    print(f"  [Evaluate] Collecting Phase B metrics...")
    phase_b_metrics = evaluate_phase_b_metrics(net, pref_deg)
    for k_m, v_m in phase_b_metrics.items():
        metrics[f"phase_b_{k_m}"] = v_m

    elapsed = time.time() - t_start
    metrics["elapsed_seconds"] = round(elapsed, 1)
    print(f"  [Done] {key} seed={seed}: "
          f"OSI={metrics.get('phase_a_osi_mean', 0):.3f}, "
          f"weight_ratio={metrics.get('phase_b_fwd_rev_weight_ratio', 0):.3f}, "
          f"spike_ratio={metrics.get('phase_b_spike_ratio', 0):.3f}, "
          f"omission={metrics.get('phase_b_omission_signal', 0):.4f} "
          f"({elapsed:.0f}s)")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def _build_base_checkpoint(
    seed: int,
) -> Tuple[RgcLgnV1Network, Dict[str, Any]]:
    """Build baseline network through Phase A + calibration. Returns (net, phase_a_metrics)."""
    print(f"\n  [Base checkpoint] Building for seed={seed}...")
    p = Params(**{**BASE_PARAMS, "seed": seed})
    net = RgcLgnV1Network(p, init_mode="random")

    print(f"  [Base Phase A] {PHASE_A_SEGMENTS} segments...")
    run_phase_a(net, PHASE_A_SEGMENTS)

    pa_metrics = evaluate_phase_a_metrics(net)
    stability_a = evaluate_stability(net)
    pa_metrics.update({f"stability_a_{k}": v for k, v in stability_a.items()})

    print(f"  [Base Phase A] OSI={pa_metrics['osi_mean']:.3f}, "
          f"n_tuned={pa_metrics['n_tuned']}/{net.M}, "
          f"rate={pa_metrics['mean_rate_hz']:.2f} Hz")

    print(f"  [Base Calibrate] Target E→E drive = {TARGET_EE_DRIVE:.2f}...")
    scale, frac = calibrate_ee_drive(net, TARGET_EE_DRIVE, contrast=CONTRAST)
    off_diag = net.W_e_e[net.mask_e_e]
    print(f"  [Base Calibrate] scale={scale:.1f}, frac={frac:.4f}, "
          f"W_ee mean={off_diag.mean():.5f}")
    pa_metrics["ee_calibration_scale"] = float(scale)
    pa_metrics["ee_calibration_frac"] = float(frac)
    pa_metrics["ee_calibration_mean"] = float(off_diag.mean())

    return net, pa_metrics


def run_all(
    keys: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    out_dir: str = OUT_DIR,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run all experiments across seeds. Returns {key: [metrics_per_seed]}.

    Checkpoint optimization: for each seed, Phase A + calibration is run once
    with baseline params. BL and all Category B experiments share this checkpoint.
    Category A (non-BL) and C experiments build fresh networks with modified params.
    """
    if keys is None:
        keys = list(EXPERIMENTS.keys())
    if seeds is None:
        seeds = list(DEFAULT_SEEDS)

    os.makedirs(out_dir, exist_ok=True)

    # Partition experiments
    bl_requested = "BL" in keys
    cat_b_keys = [k for k in keys if EXPERIMENTS[k]["category"] == "B"]
    cat_ac_keys = [k for k in keys
                   if EXPERIMENTS[k]["category"] in ("A", "C") and k != "BL"]
    need_base_checkpoint = bl_requested or len(cat_b_keys) > 0

    # Order: base checkpoint → BL → Category B → Category A (non-BL) / C
    all_results: Dict[str, List[Dict[str, Any]]] = {k: [] for k in keys}
    total_runs = len(keys) * len(seeds)
    run_count = 0

    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"# SEED {seed}")
        print(f"{'#'*70}")

        # ── Build shared baseline checkpoint (once per seed) ──
        # Skip if all experiments that would use it already have cached results.
        base_checkpoint: Optional[Tuple[RgcLgnV1Network, Dict[str, Any]]] = None
        if need_base_checkpoint:
            checkpoint_users = (["BL"] if bl_requested else []) + cat_b_keys
            any_uncached = any(
                not os.path.exists(os.path.join(out_dir, k, f"seed_{seed}.json"))
                for k in checkpoint_users
            )
            if any_uncached:
                base_checkpoint = _build_base_checkpoint(seed)
            else:
                print(f"\n  [Base checkpoint] All BL/B experiments cached for seed={seed}, skipping")

        # ── BL: run Phase B on a deep copy of the checkpoint ──
        if bl_requested:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}]", end="")

            exp_dir = os.path.join(out_dir, "BL")
            os.makedirs(exp_dir, exist_ok=True)
            result_path = os.path.join(exp_dir, f"seed_{seed}.json")

            if os.path.exists(result_path):
                print(f"  BL seed={seed} already exists, loading...")
                with open(result_path, "r") as f:
                    all_results["BL"].append(json.load(f))
            else:
                try:
                    m = run_single_experiment(
                        "BL", seed, baseline_checkpoint=base_checkpoint,
                    )
                except Exception as e:
                    print(f"\n  ERROR in BL seed={seed}: {e}")
                    traceback.print_exc()
                    m = {"key": "BL", "seed": seed, "error": str(e)}
                all_results["BL"].append(m)
                _save_json(m, result_path)

        # ── Category B: run Phase B on deep copies of the checkpoint ──
        for key in cat_b_keys:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}]", end="")

            exp_dir = os.path.join(out_dir, key)
            os.makedirs(exp_dir, exist_ok=True)
            result_path = os.path.join(exp_dir, f"seed_{seed}.json")

            if os.path.exists(result_path):
                print(f"  {key} seed={seed} already exists, loading...")
                with open(result_path, "r") as f:
                    all_results[key].append(json.load(f))
                continue

            try:
                m = run_single_experiment(
                    key, seed, baseline_checkpoint=base_checkpoint,
                )
            except Exception as e:
                print(f"\n  ERROR in {key} seed={seed}: {e}")
                traceback.print_exc()
                m = {"key": key, "seed": seed, "error": str(e)}

            all_results[key].append(m)
            _save_json(m, result_path)

        # Release checkpoint memory before Category A/C
        base_checkpoint = None

        # ── Category A (non-BL) and C: fresh network with modified params ──
        for key in cat_ac_keys:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}]", end="")

            exp_dir = os.path.join(out_dir, key)
            os.makedirs(exp_dir, exist_ok=True)
            result_path = os.path.join(exp_dir, f"seed_{seed}.json")

            if os.path.exists(result_path):
                print(f"  {key} seed={seed} already exists, loading...")
                with open(result_path, "r") as f:
                    all_results[key].append(json.load(f))
                continue

            try:
                m = run_single_experiment(key, seed)
            except Exception as e:
                print(f"\n  ERROR in {key} seed={seed}: {e}")
                traceback.print_exc()
                m = {"key": key, "seed": seed, "error": str(e)}

            all_results[key].append(m)
            _save_json(m, result_path)

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Statistical comparison
# ──────────────────────────────────────────────────────────────────────────────
def paired_ttest(
    baseline_vals: List[float],
    ablation_vals: List[float],
) -> Dict[str, float]:
    """Paired t-test (two-sided). Returns t-statistic, p-value, effect size (Cohen's d)."""
    bl = np.array(baseline_vals, dtype=np.float64)
    ab = np.array(ablation_vals, dtype=np.float64)
    n = min(len(bl), len(ab))
    if n < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0, "n": n}

    bl, ab = bl[:n], ab[:n]
    diff = bl - ab
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))

    if std_diff < 1e-12:
        t_stat = 0.0 if abs(mean_diff) < 1e-12 else float(np.sign(mean_diff)) * 1e6
    else:
        t_stat = mean_diff / (std_diff / math.sqrt(n))

    # Two-sided p-value from t-distribution (approximate via normal for n>=5)
    # Use scipy if available, otherwise normal approximation
    try:
        from scipy import stats as sp_stats
        p_value = float(sp_stats.t.sf(abs(t_stat), df=n - 1) * 2)
    except ImportError:
        # Normal approximation (adequate for |t| > 2 or n >= 5)
        z = abs(t_stat)
        p_value = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))

    pooled_std = float(np.sqrt(0.5 * (bl.var(ddof=1) + ab.var(ddof=1))))
    cohens_d = mean_diff / pooled_std if pooled_std > 1e-12 else 0.0

    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "cohens_d": round(cohens_d, 4),
        "n": n,
        "mean_diff": round(mean_diff, 6),
    }


def classify_effect(
    p_value: float,
    metric_delta: float,
    threshold: float,
) -> str:
    """Classify as Matters / Namesake / Harmful."""
    if p_value > 0.1 or abs(metric_delta) < threshold:
        return "Namesake"
    if metric_delta < -threshold and p_value < 0.05:
        return "Harmful"  # ablation improved the metric
    if p_value < 0.05:
        return "Matters"
    return "Namesake"


def statistical_comparison(
    all_results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Compare each ablation to BL across seeds. Returns comparison dict."""
    bl_runs = all_results.get("BL", [])
    if not bl_runs:
        print("WARNING: No baseline (BL) results for statistical comparison.")
        return {}

    # Key metrics and their thresholds for "meaningful" effect
    metric_configs = [
        ("phase_a_osi_mean", 0.05),
        ("phase_b_fwd_rev_weight_ratio", 0.2),
        ("phase_b_spike_ratio", 0.05),
        ("phase_b_omission_signal", 0.001),
        ("phase_b_osi_post", 0.05),
    ]

    comparisons: Dict[str, Dict[str, Any]] = {}

    for key, runs in all_results.items():
        if key == "BL":
            continue
        if not runs or "error" in runs[0]:
            continue

        comp: Dict[str, Any] = {"key": key, "name": EXPERIMENTS[key]["name"]}

        for metric_name, threshold in metric_configs:
            bl_vals = [r.get(metric_name, float("nan")) for r in bl_runs]
            ab_vals = [r.get(metric_name, float("nan")) for r in runs]

            # Filter NaN
            valid = [
                (b, a) for b, a in zip(bl_vals, ab_vals)
                if not (math.isnan(b) or math.isnan(a))
            ]
            if len(valid) < 2:
                comp[metric_name] = {"verdict": "Insufficient data"}
                continue

            bl_valid = [v[0] for v in valid]
            ab_valid = [v[1] for v in valid]

            stats = paired_ttest(bl_valid, ab_valid)
            bl_mean = float(np.mean(bl_valid))
            ab_mean = float(np.mean(ab_valid))
            delta = bl_mean - ab_mean  # positive means ablation reduced the metric
            verdict = classify_effect(stats["p_value"], delta, threshold)

            comp[metric_name] = {
                "bl_mean": round(bl_mean, 5),
                "ab_mean": round(ab_mean, 5),
                "delta": round(delta, 5),
                **stats,
                "verdict": verdict,
            }

        # Overall verdict: "Matters" if any key metric matters
        verdicts = [
            comp.get(m, {}).get("verdict", "Namesake")
            for m, _ in metric_configs
        ]
        if "Matters" in verdicts:
            comp["overall_verdict"] = "Matters"
        elif "Harmful" in verdicts:
            comp["overall_verdict"] = "Harmful"
        else:
            comp["overall_verdict"] = "Namesake"

        comparisons[key] = comp

    return comparisons


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────
def generate_summary_table(
    all_results: Dict[str, List[Dict[str, Any]]],
    comparisons: Dict[str, Dict[str, Any]],
    out_path: str,
) -> None:
    """Write summary_table.md with a comparison table."""
    lines: List[str] = []
    lines.append("# Ablation Study: Summary Table\n")
    lines.append(
        "| ID | Component | Category | OSI (mean) | Wt Ratio (fwd/rev) | "
        "Spike Ratio | Omission | OSI post | Verdict |"
    )
    lines.append(
        "|:---|:----------|:---------|:-----------|:--------------------|:"
        "-----------|:---------|:---------|:--------|"
    )

    for key in EXPERIMENTS:
        runs = all_results.get(key, [])
        if not runs:
            continue

        exp = EXPERIMENTS[key]
        name = exp["name"]
        cat = exp["category"]

        # Aggregate means across seeds
        def _mean(metric: str) -> str:
            vals = [r.get(metric, float("nan")) for r in runs if "error" not in r]
            vals = [v for v in vals if not math.isnan(v)]
            if not vals:
                return "—"
            return f"{np.mean(vals):.3f}"

        osi_a = _mean("phase_a_osi_mean")
        wt_ratio = _mean("phase_b_fwd_rev_weight_ratio")
        spk_ratio = _mean("phase_b_spike_ratio")
        omission = _mean("phase_b_omission_signal")
        osi_post = _mean("phase_b_osi_post")

        if key == "BL":
            verdict = "**Baseline**"
        elif key in comparisons:
            verdict = f"**{comparisons[key].get('overall_verdict', '—')}**"
        else:
            verdict = "—"

        lines.append(
            f"| {key} | {name} | {cat} | {osi_a} | {wt_ratio} | "
            f"{spk_ratio} | {omission} | {osi_post} | {verdict} |"
        )

    lines.append("")
    lines.append(f"Seeds: {DEFAULT_SEEDS}, M={M_DEFAULT}, N={N_DEFAULT}")
    lines.append(f"Phase A: {PHASE_A_SEGMENTS} segments, Phase B: {PHASE_B_PRESENTATIONS} presentations")
    lines.append(f"STDP: additive (A+={BASE_PARAMS['ee_stdp_A_plus']}, "
                 f"A-={BASE_PARAMS['ee_stdp_A_minus']})")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Summary table written to {out_path}")


def generate_report(
    all_results: Dict[str, List[Dict[str, Any]]],
    comparisons: Dict[str, Dict[str, Any]],
    out_path: str,
) -> None:
    """Write detailed summary_report.md with per-component analysis."""
    lines: List[str] = []
    lines.append("# Ablation Study: Detailed Report\n")
    lines.append("## Overview\n")
    lines.append(
        "This report presents results of systematic component ablation "
        "experiments on the RGC→LGN→V1 spiking network. Each component "
        "was disabled independently (or in combination) to assess its "
        "contribution to orientation selectivity (OSI) and sequence learning.\n"
    )
    lines.append("### Protocol\n")
    lines.append(f"- **Seeds**: {DEFAULT_SEEDS}")
    lines.append(f"- **Network**: M={M_DEFAULT} ensembles, N={N_DEFAULT} patch size")
    lines.append(f"- **Phase A**: {PHASE_A_SEGMENTS} segments, golden-ratio orientation sweep")
    lines.append(f"- **Phase B**: {PHASE_B_PRESENTATIONS} sequence presentations "
                 f"(ABCD = {SEQ_THETAS}°)")
    lines.append(f"- **STDP mode**: Additive (A+={BASE_PARAMS['ee_stdp_A_plus']}, "
                 f"A-={BASE_PARAMS['ee_stdp_A_minus']})")
    lines.append(f"- **E→E drive target**: {TARGET_EE_DRIVE}")
    lines.append(f"- **Contrast**: {CONTRAST}")
    lines.append("")

    # ── Baseline summary ──
    bl_runs = all_results.get("BL", [])
    if bl_runs:
        lines.append("## Baseline (BL)\n")
        for metric_name in [
            "phase_a_osi_mean", "phase_a_n_tuned", "phase_a_mean_rate_hz",
            "phase_b_fwd_rev_weight_ratio", "phase_b_spike_ratio",
            "phase_b_omission_signal", "phase_b_osi_post",
        ]:
            vals = [r.get(metric_name, float("nan")) for r in bl_runs if "error" not in r]
            vals = [v for v in vals if not math.isnan(v)]
            if vals:
                lines.append(
                    f"- **{metric_name}**: {np.mean(vals):.4f} "
                    f"± {np.std(vals):.4f} (n={len(vals)})"
                )
        lines.append("")

    # ── Per-component sections ──
    lines.append("---\n")
    lines.append("## Per-Component Results\n")

    for key in EXPERIMENTS:
        if key == "BL":
            continue

        exp = EXPERIMENTS[key]
        runs = all_results.get(key, [])
        comp = comparisons.get(key, {})

        lines.append(f"### {key}: {exp['name']}\n")
        lines.append(f"**Category**: {exp['category']}")
        lines.append(f"**Biology**: {exp['bio_description']}")
        lines.append(f"**Hypothesis**: {exp['hypothesis']}\n")

        if not runs:
            lines.append("*No results available.*\n")
            continue

        # Per-seed summary
        lines.append("**Per-seed results:**\n")
        lines.append(
            "| Seed | OSI | Wt Ratio | Spike Ratio | Omission | OSI post |"
        )
        lines.append(
            "|:-----|:----|:---------|:------------|:---------|:---------|"
        )
        for r in runs:
            if "error" in r:
                lines.append(f"| {r.get('seed', '?')} | ERROR: {r['error']} |||||")
                continue
            lines.append(
                f"| {r.get('seed', '?')} "
                f"| {r.get('phase_a_osi_mean', 0):.3f} "
                f"| {r.get('phase_b_fwd_rev_weight_ratio', 0):.3f} "
                f"| {r.get('phase_b_spike_ratio', 0):.3f} "
                f"| {r.get('phase_b_omission_signal', 0):.4f} "
                f"| {r.get('phase_b_osi_post', 0):.3f} |"
            )

        # Statistical comparison
        if comp:
            lines.append("\n**Statistical comparison vs. baseline:**\n")
            for metric_name in [
                "phase_a_osi_mean", "phase_b_fwd_rev_weight_ratio",
                "phase_b_spike_ratio", "phase_b_omission_signal", "phase_b_osi_post",
            ]:
                mc = comp.get(metric_name, {})
                if isinstance(mc, dict) and "verdict" in mc:
                    if mc["verdict"] == "Insufficient data":
                        lines.append(f"- **{metric_name}**: Insufficient data")
                    else:
                        lines.append(
                            f"- **{metric_name}**: BL={mc.get('bl_mean', '?'):.4f} vs "
                            f"ablation={mc.get('ab_mean', '?'):.4f} "
                            f"(Δ={mc.get('delta', 0):.4f}, t={mc.get('t_stat', 0):.2f}, "
                            f"p={mc.get('p_value', 1):.4f}, d={mc.get('cohens_d', 0):.2f}) "
                            f"→ **{mc['verdict']}**"
                        )

            overall = comp.get("overall_verdict", "—")
            lines.append(f"\n**Overall verdict: {overall}**\n")
        else:
            lines.append("\n*No statistical comparison available.*\n")

        lines.append("---\n")

    # ── Summary of verdicts ──
    lines.append("## Summary of Verdicts\n")
    lines.append("| Component | Verdict | Key Evidence |")
    lines.append("|:----------|:--------|:-------------|")

    for key in EXPERIMENTS:
        if key == "BL":
            continue
        exp = EXPERIMENTS[key]
        comp = comparisons.get(key, {})
        verdict = comp.get("overall_verdict", "—")

        # Find the most significant metric
        best_metric = "—"
        best_p = 1.0
        for metric_name in [
            "phase_a_osi_mean", "phase_b_fwd_rev_weight_ratio",
            "phase_b_spike_ratio", "phase_b_omission_signal",
        ]:
            mc = comp.get(metric_name, {})
            if isinstance(mc, dict) and mc.get("p_value", 1.0) < best_p:
                best_p = mc["p_value"]
                best_metric = f"{metric_name}: p={best_p:.4f}"

        lines.append(f"| {key}: {exp['name']} | **{verdict}** | {best_metric} |")

    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Detailed report written to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _save_json(data: Dict[str, Any], path: str) -> None:
    """Save dict to JSON, handling numpy types."""
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(data), f, indent=2)


def load_existing_results(out_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load all existing per-seed JSON results from out_dir."""
    results: Dict[str, List[Dict[str, Any]]] = {}
    for key in EXPERIMENTS:
        exp_dir = os.path.join(out_dir, key)
        if not os.path.isdir(exp_dir):
            continue
        runs = []
        for fname in sorted(os.listdir(exp_dir)):
            if fname.startswith("seed_") and fname.endswith(".json"):
                with open(os.path.join(exp_dir, fname), "r") as f:
                    runs.append(json.load(f))
        if runs:
            results[key] = runs
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Systematic ablation study for RGC-LGN-V1 network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--keys", nargs="*", default=None,
        help="Experiment keys to run (default: all). E.g., --keys BL A1 A5 B10",
    )
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=None,
        help="Seeds to use (default: 1 2 3 4 5). E.g., --seeds 1 2 3",
    )
    parser.add_argument(
        "--out", type=str, default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick verification: single seed (1), BL + A5 only",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip experiments, just regenerate reports from existing results",
    )

    args = parser.parse_args()

    if args.quick:
        args.keys = ["BL", "A5"]
        args.seeds = [1]
        print("[Quick mode] Running BL + A5 with seed=1 only")

    keys = args.keys
    seeds = args.seeds
    out_dir = args.out

    # Validate keys
    if keys is not None:
        for k in keys:
            if k not in EXPERIMENTS:
                print(f"ERROR: Unknown experiment key '{k}'. "
                      f"Valid keys: {list(EXPERIMENTS.keys())}")
                return 1

    t_total_start = time.time()

    if not args.report_only:
        # Run experiments
        all_results = run_all(keys=keys, seeds=seeds, out_dir=out_dir)
    else:
        all_results = load_existing_results(out_dir)
        if not all_results:
            print(f"ERROR: No results found in {out_dir}/")
            return 1
        print(f"Loaded results for {list(all_results.keys())}")

    # Statistical comparison
    print(f"\n{'='*70}")
    print(f"STATISTICAL ANALYSIS")
    print(f"{'='*70}")
    comparisons = statistical_comparison(all_results)

    for key, comp in comparisons.items():
        verdict = comp.get("overall_verdict", "—")
        print(f"  {key:4s} ({EXPERIMENTS[key]['name']:30s}): {verdict}")

    # Generate reports
    print(f"\nGenerating reports...")
    generate_summary_table(
        all_results, comparisons,
        os.path.join(out_dir, "summary_table.md"),
    )
    generate_report(
        all_results, comparisons,
        os.path.join(out_dir, "summary_report.md"),
    )

    # Save raw comparisons
    _save_json(comparisons, os.path.join(out_dir, "comparisons.json"))

    elapsed_total = time.time() - t_total_start
    print(f"\nTotal time: {elapsed_total / 60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

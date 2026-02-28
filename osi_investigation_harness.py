#!/usr/bin/env python3
"""Shared investigation harness for OSI mechanism investigation.

Provides standardised infrastructure for running ablation, dose-response,
and mechanistic-pathway experiments across the RGC-LGN-V1 simulation.

Every per-mechanism script (investigate_M1.py … investigate_M6.py) imports
from this module so that metrics, checkpointing, and plotting are consistent.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the simulation module is importable
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from biologically_plausible_v1_stdp import (
    Params,
    RgcLgnV1Network,
    compute_osi,
    tuning_summary,
    onoff_weight_corr,
    rf_fft_orientation_metrics,
    circ_mean_resultant_180,
    max_circ_gap_180,
)

# ---------------------------------------------------------------------------
# Matplotlib — use non-interactive backend so agents never block on display
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class InvestigationConfig:
    """Configuration for one investigation condition."""
    condition_name: str
    param_overrides: Dict[str, Any] = field(default_factory=dict)
    seeds: List[int] = field(default_factory=lambda: [1, 42, 137])
    M: int = 16
    N: int = 8
    train_segments: int = 300
    segment_ms: int = 300
    checkpoint_every: int = 10
    eval_thetas: int = 12          # number of orientations for tuning eval
    eval_repeats: int = 3          # repeats per orientation
    eval_contrast: float = 1.0
    train_contrast: float = 1.0
    out_dir: str = "investigation_results"

    @property
    def thetas_deg(self) -> np.ndarray:
        return np.linspace(0, 180, self.eval_thetas, endpoint=False)


# ===========================================================================
# Weight / tuning metrics computed at each checkpoint
# ===========================================================================

def gini_coefficient(arr: np.ndarray) -> float:
    """Gini coefficient of a 1-D array (0 = perfectly equal, 1 = maximally unequal)."""
    a = np.sort(np.abs(arr).ravel()).astype(np.float64)
    n = a.size
    if n == 0 or a.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (index * a).sum() / (n * a.sum())) - (n + 1.0) / n)


def weight_entropy(W_row: np.ndarray, n_bins: int = 30) -> float:
    """Shannon entropy of normalised weight histogram for one neuron."""
    w = W_row.ravel().astype(np.float64)
    if w.max() - w.min() < 1e-12:
        return 0.0
    hist, _ = np.histogram(w, bins=n_bins, range=(0, float(w.max()) + 1e-9))
    p = hist.astype(np.float64) / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_weight_metrics(net: RgcLgnV1Network) -> Dict[str, Any]:
    """Standard metrics snapshot from the current network state.

    Returns a dict whose values are JSON-serialisable (lists/floats).
    """
    p = net.p
    W = net.W.copy()
    M = net.M
    N = net.N
    n_pix = N * N
    thetas = np.linspace(0, 180, 12, endpoint=False)

    # --- Spike-based OSI via evaluate_tuning ---
    rates = net.evaluate_tuning(thetas, repeats=3)
    osi, pref = compute_osi(rates, thetas)
    ts = tuning_summary(rates, thetas)

    # --- Weight-based RF metrics ---
    kwargs: Dict[str, Any] = {}
    if hasattr(net, "X_on") and net.X_on is not None:
        kwargs.update(X_on=net.X_on, Y_on=net.Y_on,
                      X_off=net.X_off, Y_off=net.Y_off)
    on_to_off = getattr(net, "on_to_off", None)

    oo_corr = onoff_weight_corr(W, N, on_to_off=on_to_off, **kwargs)
    rf_orient, rf_pref = rf_fft_orientation_metrics(W, N, on_to_off=on_to_off, **kwargs)

    # --- Per-ensemble weight statistics ---
    W_on = W[:, :n_pix]
    W_off = W[:, n_pix:]
    w_total = W.sum(axis=1)
    w_on_total = W_on.sum(axis=1)
    w_off_total = W_off.sum(axis=1)
    w_gini = np.array([gini_coefficient(W[m]) for m in range(M)])
    sat_frac = (W > 0.9 * p.w_max).sum(axis=1) / W.shape[1]  # fraction near ceiling
    w_ent = np.array([weight_entropy(W[m]) for m in range(M)])

    # --- Population diversity ---
    R, mu = circ_mean_resultant_180(pref)
    gap = max_circ_gap_180(pref)

    return {
        # Tuning
        "osi": osi.tolist(),
        "pref_deg": pref.tolist(),
        "peak_rate_hz": ts["peak_rate_hz"].tolist(),
        "mean_osi": float(np.mean(osi)),
        "mean_rate_hz": float(np.mean(ts["peak_rate_hz"])),
        # Weight totals
        "w_total": w_total.tolist(),
        "w_on_total": w_on_total.tolist(),
        "w_off_total": w_off_total.tolist(),
        # Weight structure
        "onoff_corr": oo_corr.tolist(),
        "w_gini": w_gini.tolist(),
        "sat_frac": sat_frac.tolist(),
        "w_entropy": w_ent.tolist(),
        # RF-based orientation
        "rf_orientedness": rf_orient.tolist(),
        "rf_pref_deg": rf_pref.tolist(),
        # Population diversity
        "pref_resultant_R": float(R),
        "pref_mean_deg": float(mu),
        "pref_max_gap": float(gap),
        # Raw rates for downstream analysis
        "rates_hz": rates.tolist(),
    }


# ===========================================================================
# Longitudinal trajectory analysis (post-hoc, from checkpoint timeseries)
# ===========================================================================

def compute_longitudinal_trajectory(checkpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive longitudinal dynamics from a list of checkpoint metric dicts.

    Parameters
    ----------
    checkpoints : list of dicts
        Each element is the output of ``compute_weight_metrics`` at successive
        training checkpoints.  Must contain at least ``osi`` and ``pref_deg``.

    Returns
    -------
    dict with keys:
        segments       : list of segment indices for each checkpoint
        osi_trajectory : (n_ckpt, M) OSI over time
        pref_trajectory: (n_ckpt, M) preferred orientation over time
        convergence_seg: (M,) first segment where OSI > 0.3 AND pref within 15 deg of final
        flip_count     : (M,) number of pref changes > 30 deg between consecutive checkpoints
    """
    n = len(checkpoints)
    if n == 0:
        return {}
    M = len(checkpoints[0]["osi"])

    osi_traj = np.array([c["osi"] for c in checkpoints])           # (n, M)
    pref_traj = np.array([c["pref_deg"] for c in checkpoints])     # (n, M)
    segments = [c.get("segment", i) for i, c in enumerate(checkpoints)]

    # Convergence: first checkpoint where OSI > 0.3 AND pref within 15 deg of final
    final_pref = pref_traj[-1]  # (M,)
    convergence_seg = np.full(M, -1, dtype=np.int32)
    for m in range(M):
        for t in range(n):
            dpref = abs(((pref_traj[t, m] - final_pref[m] + 90) % 180) - 90)
            if osi_traj[t, m] > 0.3 and dpref < 15.0:
                convergence_seg[m] = segments[t]
                break

    # Flip count: pref changes > 30 deg between consecutive checkpoints
    flip_count = np.zeros(M, dtype=np.int32)
    for t in range(1, n):
        dpref = np.abs(((pref_traj[t] - pref_traj[t - 1] + 90) % 180) - 90)
        flip_count += (dpref > 30).astype(np.int32)

    return {
        "segments": segments,
        "osi_trajectory": osi_traj.tolist(),
        "pref_trajectory": pref_traj.tolist(),
        "convergence_seg": convergence_seg.tolist(),
        "flip_count": flip_count.tolist(),
    }


# ===========================================================================
# Core investigation runner
# ===========================================================================

def _make_params(cfg: InvestigationConfig, seed: int) -> Params:
    """Construct Params with investigation defaults + overrides."""
    kw: Dict[str, Any] = dict(
        M=cfg.M,
        N=cfg.N,
        segment_ms=cfg.segment_ms,
        train_segments=0,           # we drive training manually
        train_stimulus="grating",
        train_contrast=cfg.train_contrast,
        seed=seed,
    )
    kw.update(cfg.param_overrides)
    return Params(**kw)


def run_single_seed(
    cfg: InvestigationConfig,
    seed: int,
    *,
    extra_checkpoint_fn: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one seed of an investigation condition.

    Parameters
    ----------
    cfg : InvestigationConfig
    seed : int
    extra_checkpoint_fn : callable(net, segment) -> dict, optional
        If provided, called at each checkpoint to collect mechanism-specific
        diagnostics.  The returned dict is merged into the checkpoint record.
    verbose : bool

    Returns
    -------
    dict with keys:
        seed, condition, params (serialised), checkpoints (list of dicts),
        trajectory (longitudinal analysis), wall_time_s
    """
    t0 = time.time()
    p = _make_params(cfg, seed)
    net = RgcLgnV1Network(p)

    # Golden-ratio orientation schedule
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    theta_step = 180.0 / phi
    theta_offset = float(net.rng.uniform(0.0, 180.0))

    checkpoints: List[Dict[str, Any]] = []

    # Initial checkpoint (segment 0)
    if verbose:
        print(f"  [{cfg.condition_name}|seed={seed}] checkpoint seg=0 ...")
    ckpt = compute_weight_metrics(net)
    ckpt["segment"] = 0
    ckpt["W_snapshot"] = net.W.copy().tolist()
    if extra_checkpoint_fn is not None:
        ckpt.update(extra_checkpoint_fn(net, 0))
    checkpoints.append(ckpt)

    # Training loop with periodic checkpoints
    for s in range(1, cfg.train_segments + 1):
        th = float((theta_offset + (s - 1) * theta_step) % 180.0)
        net.run_segment(th, plastic=True, contrast=cfg.train_contrast)

        if s % cfg.checkpoint_every == 0 or s == cfg.train_segments:
            if verbose:
                print(f"  [{cfg.condition_name}|seed={seed}] checkpoint seg={s} ...")
            ckpt = compute_weight_metrics(net)
            ckpt["segment"] = s
            ckpt["W_snapshot"] = net.W.copy().tolist()
            if extra_checkpoint_fn is not None:
                ckpt.update(extra_checkpoint_fn(net, s))
            checkpoints.append(ckpt)

    trajectory = compute_longitudinal_trajectory(checkpoints)

    wall = time.time() - t0
    if verbose:
        final_osi = checkpoints[-1]["mean_osi"]
        print(f"  [{cfg.condition_name}|seed={seed}] done in {wall:.1f}s  "
              f"final_mean_osi={final_osi:.3f}")

    return {
        "seed": seed,
        "condition": cfg.condition_name,
        "params": {k: v for k, v in asdict(p).items()
                   if not isinstance(v, (np.ndarray,))},
        "checkpoints": checkpoints,
        "trajectory": trajectory,
        "wall_time_s": wall,
    }


def run_investigation(
    cfg: InvestigationConfig,
    *,
    extra_checkpoint_fn: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all seeds for one investigation condition.

    Returns
    -------
    dict with keys: condition, seeds (list of per-seed results), summary
    """
    results: List[Dict[str, Any]] = []
    for seed in cfg.seeds:
        if verbose:
            print(f"[{cfg.condition_name}] seed={seed}")
        res = run_single_seed(cfg, seed,
                              extra_checkpoint_fn=extra_checkpoint_fn,
                              verbose=verbose)
        results.append(res)

    # Aggregate summary across seeds
    final_osis = [r["checkpoints"][-1]["mean_osi"] for r in results]
    summary = {
        "condition": cfg.condition_name,
        "n_seeds": len(cfg.seeds),
        "final_mean_osi": float(np.mean(final_osis)),
        "final_sem_osi": float(np.std(final_osis) / max(1, np.sqrt(len(final_osis)))),
        "final_osi_per_seed": final_osis,
    }
    if verbose:
        print(f"[{cfg.condition_name}] SUMMARY: "
              f"mean_osi={summary['final_mean_osi']:.3f} "
              f"+/- {summary['final_sem_osi']:.3f}")

    return {
        "condition": cfg.condition_name,
        "seeds": results,
        "summary": summary,
    }


# ===========================================================================
# Save / load helpers
# ===========================================================================

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_condition_result(result: Dict[str, Any], out_dir: str) -> str:
    """Save a condition result dict to ``out_dir/<condition>/result.json``.

    W_snapshot arrays are saved as compressed .npz for efficiency;
    the JSON stores a placeholder path.

    Returns the directory path.
    """
    cond = result["condition"]
    d = os.path.join(out_dir, cond)
    os.makedirs(d, exist_ok=True)

    # Extract heavy W_snapshot arrays -> npz
    for si, seed_res in enumerate(result.get("seeds", [])):
        snapshots = {}
        for ci, ckpt in enumerate(seed_res.get("checkpoints", [])):
            key = f"seed{seed_res['seed']}_seg{ckpt.get('segment', ci)}"
            if "W_snapshot" in ckpt:
                snapshots[key] = np.array(ckpt["W_snapshot"], dtype=np.float32)
                ckpt["W_snapshot"] = f"__npz__:{key}"
        if snapshots:
            np.savez_compressed(os.path.join(d, f"W_snapshots_seed{seed_res['seed']}.npz"),
                                **snapshots)

    # Save JSON (without heavy arrays)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(result, f, cls=_NumpyEncoder, indent=1)

    return d


def load_condition_result(cond_dir: str) -> Dict[str, Any]:
    """Load a condition result from disk (inverse of save_condition_result)."""
    with open(os.path.join(cond_dir, "result.json")) as f:
        result = json.load(f)
    # Re-hydrate W_snapshot from npz files
    for seed_res in result.get("seeds", []):
        npz_path = os.path.join(cond_dir, f"W_snapshots_seed{seed_res['seed']}.npz")
        if os.path.exists(npz_path):
            npz = np.load(npz_path)
            for ckpt in seed_res.get("checkpoints", []):
                ws = ckpt.get("W_snapshot", "")
                if isinstance(ws, str) and ws.startswith("__npz__:"):
                    key = ws.split(":", 1)[1]
                    if key in npz:
                        ckpt["W_snapshot"] = npz[key].tolist()
            npz.close()
    return result


# ===========================================================================
# Plotting utilities
# ===========================================================================

def plot_osi_timeseries(
    results: Dict[str, Any] | List[Dict[str, Any]],
    out_path: str,
    title: str = "OSI over training",
) -> None:
    """Plot mean OSI trajectory (+/- SEM across seeds) for one or more conditions."""
    if isinstance(results, dict):
        results = [results]

    fig, ax = plt.subplots(figsize=(8, 5))
    for res in results:
        cond = res["condition"]
        all_osi = []
        segments = None
        for seed_res in res["seeds"]:
            traj = seed_res["trajectory"]
            osi_t = np.array(traj["osi_trajectory"])  # (n_ckpt, M)
            all_osi.append(osi_t.mean(axis=1))        # mean across ensembles
            if segments is None:
                segments = np.array(traj["segments"])
        all_osi = np.array(all_osi)  # (n_seeds, n_ckpt)
        mean = all_osi.mean(axis=0)
        sem = all_osi.std(axis=0) / max(1, np.sqrt(all_osi.shape[0]))
        ax.plot(segments, mean, label=cond)
        ax.fill_between(segments, mean - sem, mean + sem, alpha=0.2)

    ax.set_xlabel("Training segment")
    ax.set_ylabel("Mean OSI")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pref_trajectories(
    result: Dict[str, Any],
    out_path: str,
    seed_idx: int = 0,
    title: str = "Preferred orientation trajectories",
) -> None:
    """Plot per-ensemble preferred orientation over training for one seed."""
    seed_res = result["seeds"][seed_idx]
    traj = seed_res["trajectory"]
    segments = np.array(traj["segments"])
    pref_t = np.array(traj["pref_trajectory"])  # (n_ckpt, M)
    M = pref_t.shape[1]

    fig, ax = plt.subplots(figsize=(10, 5))
    for m in range(M):
        c = hsv_to_rgb([m / M, 0.8, 0.8])
        ax.plot(segments, pref_t[:, m], color=c, alpha=0.7, linewidth=1.2)
    ax.set_xlabel("Training segment")
    ax.set_ylabel("Preferred orientation (deg)")
    ax.set_ylim(-5, 185)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dose_response(
    conditions: List[Dict[str, Any]],
    x_values: List[float],
    x_label: str,
    out_path: str,
    title: str = "Dose-response",
) -> None:
    """Plot mean final OSI vs a swept parameter value."""
    means = []
    sems = []
    for res in conditions:
        final_osis = [sr["checkpoints"][-1]["mean_osi"] for sr in res["seeds"]]
        means.append(float(np.mean(final_osis)))
        sems.append(float(np.std(final_osis) / max(1, np.sqrt(len(final_osis)))))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x_values, means, yerr=sems, marker="o", capsize=4)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean final OSI")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_weight_evolution(
    result: Dict[str, Any],
    out_path: str,
    seed_idx: int = 0,
    title: str = "Weight statistics over training",
) -> None:
    """Plot weight total, Gini, saturation fraction, and entropy over training."""
    seed_res = result["seeds"][seed_idx]
    ckpts = seed_res["checkpoints"]
    segments = [c["segment"] for c in ckpts]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Mean weight total
    w_total = [np.mean(c["w_total"]) for c in ckpts]
    axes[0, 0].plot(segments, w_total)
    axes[0, 0].set_ylabel("Mean total weight")
    axes[0, 0].set_title("Weight total")

    # Mean Gini
    w_gini = [np.mean(c["w_gini"]) for c in ckpts]
    axes[0, 1].plot(segments, w_gini)
    axes[0, 1].set_ylabel("Mean Gini coefficient")
    axes[0, 1].set_title("Weight sparsity (Gini)")

    # Mean saturation fraction
    sat = [np.mean(c["sat_frac"]) for c in ckpts]
    axes[1, 0].plot(segments, sat)
    axes[1, 0].set_ylabel("Mean saturation fraction")
    axes[1, 0].set_title("Weight saturation (>0.9*w_max)")

    # Mean entropy
    ent = [np.mean(c["w_entropy"]) for c in ckpts]
    axes[1, 1].plot(segments, ent)
    axes[1, 1].set_ylabel("Mean weight entropy (bits)")
    axes[1, 1].set_title("Weight distribution entropy")

    for ax in axes.flat:
        ax.set_xlabel("Training segment")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_condition_report(
    result: Dict[str, Any],
    out_path: str,
) -> None:
    """Write a concise markdown report for one condition."""
    s = result["summary"]
    lines = [
        f"# {result['condition']}",
        "",
        f"**Seeds**: {s['n_seeds']}",
        f"**Final mean OSI**: {s['final_mean_osi']:.4f} +/- {s['final_sem_osi']:.4f}",
        f"**Per-seed OSI**: {[f'{v:.3f}' for v in s['final_osi_per_seed']]}",
        "",
    ]

    # Convergence info from first seed
    if result["seeds"]:
        traj = result["seeds"][0]["trajectory"]
        if "convergence_seg" in traj:
            conv = traj["convergence_seg"]
            n_conv = sum(1 for c in conv if c >= 0)
            lines.append(f"**Ensembles converged (seed 0)**: {n_conv}/{len(conv)}")
            if n_conv > 0:
                conv_segs = [c for c in conv if c >= 0]
                lines.append(f"**Mean convergence segment**: {np.mean(conv_segs):.1f}")
        if "flip_count" in traj:
            lines.append(f"**Mean flip count**: {np.mean(traj['flip_count']):.1f}")

    lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ===========================================================================
# Convenience: run baseline
# ===========================================================================

def run_baseline(out_dir: str = "investigation_results", verbose: bool = True) -> Dict[str, Any]:
    """Run the full-model baseline and save results."""
    cfg = InvestigationConfig(
        condition_name="baseline",
        param_overrides={},
        out_dir=out_dir,
    )
    result = run_investigation(cfg, verbose=verbose)
    save_dir = save_condition_result(result, out_dir)
    if verbose:
        print(f"[baseline] saved to {save_dir}")
    return result


# ===========================================================================
# CLI entry point for quick testing
# ===========================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="OSI investigation harness — smoke test")
    ap.add_argument("--quick", action="store_true",
                    help="Quick smoke test: 1 seed, 30 segments, checkpoint every 10")
    ap.add_argument("--baseline", action="store_true",
                    help="Run full baseline (3 seeds, 300 segments)")
    args = ap.parse_args()

    if args.quick:
        print("=== Quick smoke test ===")
        cfg = InvestigationConfig(
            condition_name="smoke_test",
            seeds=[1],
            train_segments=30,
            checkpoint_every=10,
        )
        result = run_investigation(cfg, verbose=True)
        d = save_condition_result(result, cfg.out_dir)
        print(f"Saved to {d}")

        # Test plots
        plot_osi_timeseries(result, os.path.join(d, "osi_timeseries.png"))
        plot_pref_trajectories(result, os.path.join(d, "pref_trajectories.png"))
        plot_weight_evolution(result, os.path.join(d, "weight_evolution.png"))
        write_condition_report(result, os.path.join(d, "report.md"))
        print("Smoke test passed!")

    elif args.baseline:
        run_baseline()
    else:
        ap.print_help()

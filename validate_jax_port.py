#!/usr/bin/env python3
"""Validation and benchmarking for JAX-ported neural simulation.

Compares outputs of the numpy (biologically_plausible_v1_stdp.py) and JAX
(network_jax.py) implementations to verify correctness and measure speedup.

Sections:
    1. Structural tests (Izhikevich step, grating, STDP)
    2. Single-segment comparison (statistical)
    3. Multi-segment training comparison (30 seg)
    4. Benchmark (numpy vs JAX wall-clock)
    5. Extended training (optional, --extended flag)
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# ── imports from project ──────────────────────────────────────────────
from biologically_plausible_v1_stdp import Params, RgcLgnV1Network, compute_osi

# JAX (lazy so we can report import time)
t0_jax = time.perf_counter()
import jax
import jax.numpy as jnp
from network_jax import (
    numpy_net_to_jax_state,
    jax_state_to_numpy_net,
    run_segment_jax,
    evaluate_tuning_jax,
    izh_step,
    grating_on_coords,
    triplet_stdp_update,
)
t_jax_import = time.perf_counter() - t0_jax


# ── Helpers ───────────────────────────────────────────────────────────

n_pass = 0
n_fail = 0


def report(name: str, passed: bool, detail: str = ""):
    global n_pass, n_fail
    tag = "PASS" if passed else "FAIL"
    if passed:
        n_pass += 1
    else:
        n_fail += 1
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def section(title: str):
    print()
    print("-" * 60)
    print(f"  {title}")
    print("-" * 60)


# =====================================================================
# 1. Structural: Izhikevich step
# =====================================================================

def test_izh_step():
    section("1. Structural: Izhikevich neuron step")

    rng = np.random.default_rng(99)
    n = 32

    # RS-type neuron params
    a, b, c, d, v_peak, dt = 0.02, 0.2, -65.0, 8.0, 30.0, 0.5

    v_np = rng.uniform(-70, -50, n).astype(np.float32)
    u_np = (b * v_np).astype(np.float32)
    I_ext = rng.uniform(0, 25, n).astype(np.float32)

    # --- numpy path (reproduce IzhikevichPopulation.step logic) ---
    v = v_np.copy()
    u = u_np.copy()
    dt_sub = dt / 2.0
    for _ in range(2):
        v_c = np.clip(v, -100, v_peak)
        dv = (0.04 * v_c * v_c + 5.0 * v_c + 140.0 - u + I_ext) * dt_sub
        du = a * (b * v_c - u) * dt_sub
        v = v + dv
        u = u + du
    spk_np = (v >= v_peak).astype(np.uint8)
    spike_idx = spk_np.astype(bool)
    v[spike_idx] = c
    u[spike_idx] += d

    # --- JAX path ---
    v_jax, u_jax, spk_jax = izh_step(
        jnp.array(v_np), jnp.array(u_np), jnp.array(I_ext),
        a, b, c, d, v_peak, dt)
    v_jax = np.asarray(v_jax)
    u_jax = np.asarray(u_jax)
    spk_jax = np.asarray(spk_jax)

    v_ok = np.allclose(v, v_jax, rtol=1e-5, atol=1e-6)
    u_ok = np.allclose(u, u_jax, rtol=1e-5, atol=1e-6)
    spk_ok = np.array_equal(spk_np.astype(np.float32), spk_jax)

    report("v matches", v_ok,
           f"max |diff|={np.max(np.abs(v - v_jax)):.2e}" if not v_ok else "")
    report("u matches", u_ok,
           f"max |diff|={np.max(np.abs(u - u_jax)):.2e}" if not u_ok else "")
    report("spikes match", spk_ok)


# =====================================================================
# 2. Structural: Grating generation
# =====================================================================

def test_grating():
    section("2. Structural: Grating generation")

    rng = np.random.default_rng(42)
    N = 8
    X = rng.uniform(-4, 4, (N, N)).astype(np.float32)
    Y = rng.uniform(-4, 4, (N, N)).astype(np.float32)
    theta_deg = 45.0
    t_ms = 12.5
    phase = 1.23
    spatial_freq = 0.18
    temporal_freq = 8.0

    # --- numpy path ---
    import math
    th = math.radians(theta_deg)
    coord = X * math.cos(th) + Y * math.sin(th)
    g_np = np.sin(
        2.0 * math.pi * (spatial_freq * coord - temporal_freq * (t_ms / 1000.0)) + phase
    ).astype(np.float32)

    # --- JAX path ---
    g_jax = np.asarray(grating_on_coords(
        theta_deg, t_ms, phase,
        jnp.array(X), jnp.array(Y),
        spatial_freq, temporal_freq))

    ok = np.allclose(g_np, g_jax, rtol=1e-5, atol=1e-6)
    report("grating values match", ok,
           f"max |diff|={np.max(np.abs(g_np - g_jax)):.2e}" if not ok else "")


# =====================================================================
# 3. Structural: Triplet STDP update
# =====================================================================

def test_stdp():
    section("3. Structural: Triplet STDP update")

    rng = np.random.default_rng(137)
    M, n_lgn = 4, 32  # small for fast test
    n_pix = n_lgn // 2

    # Random initial traces and weights
    x_pre = rng.uniform(0, 0.5, (M, n_lgn)).astype(np.float32)
    x_pre_slow = rng.uniform(0, 0.3, (M, n_lgn)).astype(np.float32)
    x_post = rng.uniform(0, 0.5, M).astype(np.float32)
    x_post_slow = rng.uniform(0, 0.3, M).astype(np.float32)
    W = rng.uniform(0.05, 0.8, (M, n_lgn)).astype(np.float32)
    arrivals = (rng.uniform(0, 1, (M, n_lgn)) > 0.85).astype(np.float32)
    v1_spk = (rng.uniform(0, 1, M) > 0.6).astype(np.float32)

    # ON<->OFF identity mapping for this test
    on_to_off = np.arange(n_pix, dtype=np.int32)
    off_to_on = np.arange(n_pix, dtype=np.int32)

    # STDP params (use defaults from Params)
    p = Params()
    import math
    dt = p.dt_ms
    decay_pre = math.exp(-dt / p.tau_plus)
    decay_pre_slow = math.exp(-dt / p.tau_x)
    decay_post = math.exp(-dt / p.tau_minus)
    decay_post_slow = math.exp(-dt / p.tau_y)

    # --- numpy reference (replicate logic from TripletSTDP.update) ---
    xp = x_pre.copy() * decay_pre
    xps = x_pre_slow.copy() * decay_pre_slow
    xo = x_post.copy() * decay_post
    xos = x_post_slow.copy() * decay_post_slow

    dW_np = np.zeros_like(W)
    # LTD
    dW_np -= p.A2_minus * arrivals * xo[:, None] * W
    # Update pre traces
    xp += arrivals
    xps += arrivals
    # LTP
    post_mask = v1_spk
    triplet_boost = 1.0 + p.A3_plus * xos[:, None] / (p.A2_plus + 1e-30)
    dW_np += p.A2_plus * post_mask[:, None] * xp * (p.w_max - W) * triplet_boost
    # Het depression
    inactive = 1.0 - arrivals
    dW_np -= p.A_het * post_mask[:, None] * inactive * W
    # ON/OFF split
    on_trace = xp[:, :n_pix]
    off_trace = xp[:, n_pix:]
    on_at_off = on_trace[:, off_to_on]
    off_at_on = off_trace[:, on_to_off]
    dW_np[:, n_pix:] -= p.A_split * post_mask[:, None] * on_at_off * W[:, n_pix:]
    dW_np[:, :n_pix] -= p.A_split * post_mask[:, None] * off_at_on * W[:, :n_pix]
    # Update post traces
    xo += v1_spk
    xos += v1_spk

    # --- JAX path ---
    xp_j, xps_j, xo_j, xos_j, dW_j = triplet_stdp_update(
        jnp.array(x_pre), jnp.array(x_pre_slow),
        jnp.array(x_post), jnp.array(x_post_slow),
        jnp.array(arrivals), jnp.array(v1_spk), jnp.array(W),
        decay_pre, decay_pre_slow, decay_post, decay_post_slow,
        p.A2_plus, p.A3_plus, p.A2_minus, p.w_max,
        p.A_het, p.A_split,
        jnp.array(on_to_off), jnp.array(off_to_on))

    dW_j_np = np.asarray(dW_j)
    xp_j_np = np.asarray(xp_j)
    xo_j_np = np.asarray(xo_j)

    dW_ok = np.allclose(dW_np, dW_j_np, rtol=1e-4, atol=1e-6)
    xpre_ok = np.allclose(xp, xp_j_np, rtol=1e-4, atol=1e-6)
    xpost_ok = np.allclose(xo, xo_j_np, rtol=1e-4, atol=1e-6)

    report("dW matches", dW_ok,
           f"max |diff|={np.max(np.abs(dW_np - dW_j_np)):.2e}" if not dW_ok else "")
    report("x_pre traces match", xpre_ok)
    report("x_post traces match", xpost_ok)


# =====================================================================
# 4. Single-segment comparison (statistical)
# =====================================================================

def test_single_segment():
    section("4. Single-segment comparison (non-plastic)")

    p = Params(M=16, N=8, seed=42)
    net = RgcLgnV1Network(p)

    # Snapshot initial state for JAX
    state, static = numpy_net_to_jax_state(net)

    theta = 90.0

    # --- numpy ---
    counts_np = net.run_segment(theta, plastic=False, contrast=1.0)

    # --- JAX ---
    state_j, counts_j = run_segment_jax(state, static, theta, 1.0, False)
    counts_j = np.asarray(counts_j)
    counts_j.block_until_ready() if hasattr(counts_j, 'block_until_ready') else None

    total_np = int(counts_np.sum())
    total_jax = int(counts_j.sum())

    nonzero = (total_np > 0) and (total_jax > 0)
    # Allow 50% tolerance since RNGs differ
    ratio_ok = True
    if total_np > 0:
        ratio = total_jax / max(total_np, 1)
        ratio_ok = 0.5 < ratio < 2.0
    else:
        ratio_ok = total_jax < 50  # both low is fine

    report("both produce spikes", nonzero,
           f"numpy={total_np}, JAX={total_jax}")
    report("spike totals in ballpark", ratio_ok,
           f"numpy={total_np}, JAX={total_jax}, ratio={total_jax/max(total_np,1):.2f}")


# =====================================================================
# 5. Multi-segment training comparison (30 seg)
# =====================================================================

def test_multi_segment(n_segments=30):
    section(f"5. Multi-segment training comparison ({n_segments} segments)")

    p = Params(M=16, N=8, seed=1)
    rng = np.random.default_rng(p.seed)
    thetas_train = rng.uniform(0, 180, n_segments)

    # --- numpy ---
    net_np = RgcLgnV1Network(p)
    print(f"  Running {n_segments} numpy training segments...", flush=True)
    t0 = time.perf_counter()
    for i, th in enumerate(thetas_train):
        net_np.run_segment(float(th), plastic=True, contrast=1.0)
    t_np = time.perf_counter() - t0
    print(f"  numpy done in {t_np:.1f}s")

    # --- JAX ---
    net_jax = RgcLgnV1Network(Params(M=16, N=8, seed=1))
    state, static = numpy_net_to_jax_state(net_jax)
    # Use same orientation sequence
    rng_j = np.random.default_rng(1)
    thetas_jax = rng_j.uniform(0, 180, n_segments)

    print(f"  Running {n_segments} JAX training segments...", flush=True)
    t0 = time.perf_counter()
    for i, th in enumerate(thetas_jax):
        state, _counts = run_segment_jax(state, static, float(th), 1.0, True)
    # Block until last computation done
    jax.block_until_ready(state.W)
    t_jax = time.perf_counter() - t0
    print(f"  JAX done in {t_jax:.1f}s")

    # --- Compare weight statistics ---
    W_np = net_np.W
    W_jax = np.asarray(state.W)

    mean_np, mean_jax = float(W_np.mean()), float(W_jax.mean())
    std_np, std_jax = float(W_np.std()), float(W_jax.std())

    mean_ok = abs(mean_np - mean_jax) / (abs(mean_np) + 1e-9) < 0.10
    std_ok = abs(std_np - std_jax) / (abs(std_np) + 1e-9) < 0.10

    report("W mean agrees within 10%", mean_ok,
           f"numpy={mean_np:.4f}, JAX={mean_jax:.4f}")
    report("W std agrees within 10%", std_ok,
           f"numpy={std_np:.4f}, JAX={std_jax:.4f}")

    # --- Evaluate tuning (OSI) ---
    thetas_eval = np.arange(0, 180, 10, dtype=np.float64)

    print("  Evaluating numpy tuning...", flush=True)
    rates_np = net_np.evaluate_tuning(thetas_eval, repeats=2, contrast=1.0)
    osi_np, _ = compute_osi(rates_np, thetas_eval)

    print("  Evaluating JAX tuning...", flush=True)
    rates_jax = evaluate_tuning_jax(state, static, thetas_eval, repeats=2, contrast=1.0)
    osi_jax, _ = compute_osi(rates_jax, thetas_eval)

    mean_osi_np = float(osi_np.mean())
    mean_osi_jax = float(osi_jax.mean())

    # Tolerance 0.10 rather than 0.05: numpy vs JAX use different PRNGs,
    # so stochastic training trajectories diverge.  Both should still produce
    # strong selectivity, but exact OSI can differ by ~0.10 after only 30 segs.
    osi_close = abs(mean_osi_np - mean_osi_jax) < 0.10
    osi_np_ok = mean_osi_np > 0.3
    osi_jax_ok = mean_osi_jax > 0.3

    report("mean OSI agrees within 0.10", osi_close,
           f"numpy={mean_osi_np:.3f}, JAX={mean_osi_jax:.3f}")
    report("numpy OSI > 0.3 (selectivity emerging)", osi_np_ok,
           f"OSI={mean_osi_np:.3f}")
    report("JAX OSI > 0.3 (selectivity emerging)", osi_jax_ok,
           f"OSI={mean_osi_jax:.3f}")


# =====================================================================
# 6. Benchmark
# =====================================================================

def benchmark(n_segments=30):
    section(f"6. Benchmark ({n_segments} segments, plastic)")

    p = Params(M=16, N=8, seed=1)
    rng = np.random.default_rng(999)
    thetas = rng.uniform(0, 180, n_segments + 3)  # 3 warmup + n_segments timed

    # --- Warmup (both) ---
    print("  Warming up numpy (3 segments)...", flush=True)
    net_np = RgcLgnV1Network(p)
    for th in thetas[:3]:
        net_np.run_segment(float(th), plastic=True, contrast=1.0)

    print("  Warming up JAX (3 segments, includes JIT compile)...", flush=True)
    net_jax = RgcLgnV1Network(Params(M=16, N=8, seed=1))
    state, static = numpy_net_to_jax_state(net_jax)
    for th in thetas[:3]:
        state, _ = run_segment_jax(state, static, float(th), 1.0, True)
    jax.block_until_ready(state.W)

    # --- Timed numpy ---
    print(f"  Timing numpy ({n_segments} segments)...", flush=True)
    net_np2 = RgcLgnV1Network(Params(M=16, N=8, seed=1))
    t0 = time.perf_counter()
    for th in thetas[3:]:
        net_np2.run_segment(float(th), plastic=True, contrast=1.0)
    t_np = time.perf_counter() - t0

    # --- Timed JAX (JIT already cached) ---
    # IMPORTANT: reuse the same `static` so the closure-based JIT cache hits.
    # Creating a new StaticConfig would trigger recompilation (different id()).
    print(f"  Timing JAX ({n_segments} segments)...", flush=True)
    net_jax2 = RgcLgnV1Network(Params(M=16, N=8, seed=1))
    state2, _static2 = numpy_net_to_jax_state(net_jax2)
    t0 = time.perf_counter()
    for th in thetas[3:]:
        state2, _ = run_segment_jax(state2, static, float(th), 1.0, True)
    jax.block_until_ready(state2.W)
    t_jax = time.perf_counter() - t0

    per_seg_np = t_np / n_segments
    per_seg_jax = t_jax / n_segments
    speedup = t_np / max(t_jax, 1e-9)

    print()
    print(f"  {'':20s} {'Total (s)':>10s}  {'Per-seg (s)':>12s}")
    print(f"  {'Numpy':20s} {t_np:10.2f}  {per_seg_np:12.4f}")
    print(f"  {'JAX (cached JIT)':20s} {t_jax:10.2f}  {per_seg_jax:12.4f}")
    print(f"  {'Speedup':20s} {speedup:10.1f}x")


# =====================================================================
# 7. Extended training (optional)
# =====================================================================

def extended_training(n_segments=300):
    section(f"7. Extended JAX training ({n_segments} segments)")

    p = Params(M=16, N=8, seed=1)
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)

    rng = np.random.default_rng(p.seed)
    thetas = rng.uniform(0, 180, n_segments)

    print(f"  Running {n_segments} JAX plastic segments...", flush=True)
    t0 = time.perf_counter()
    for i, th in enumerate(thetas):
        state, _ = run_segment_jax(state, static, float(th), 1.0, True)
        if (i + 1) % 50 == 0:
            jax.block_until_ready(state.W)
            elapsed = time.perf_counter() - t0
            print(f"    segment {i+1}/{n_segments} ({elapsed:.1f}s elapsed)", flush=True)
    jax.block_until_ready(state.W)
    t_total = time.perf_counter() - t0
    print(f"  Training done in {t_total:.1f}s ({t_total/n_segments:.3f}s/seg)")

    # Evaluate OSI
    thetas_eval = np.arange(0, 180, 10, dtype=np.float64)
    print("  Evaluating tuning...", flush=True)
    rates = evaluate_tuning_jax(state, static, thetas_eval, repeats=2, contrast=1.0)
    osi, pref = compute_osi(rates, thetas_eval)
    mean_osi = float(osi.mean())

    W_jax = np.asarray(state.W)
    print(f"  Mean OSI = {mean_osi:.3f}")
    print(f"  W mean = {W_jax.mean():.4f}, std = {W_jax.std():.4f}")

    # Known baseline: ~0.806 ± 0.02 (though exact value depends on RNG path)
    osi_baseline_ok = mean_osi > 0.5
    report("extended OSI > 0.5 (strong selectivity)", osi_baseline_ok,
           f"OSI={mean_osi:.3f}")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate JAX port of neural simulation against numpy reference.")
    parser.add_argument("--extended", action="store_true",
                        help="Run extended 300-segment JAX training")
    args = parser.parse_args()

    print("=" * 60)
    print("  JAX Port Validation Suite")
    print("=" * 60)
    print(f"  JAX version : {jax.__version__}")
    print(f"  JAX devices : {jax.devices()}")
    print(f"  JAX import  : {t_jax_import:.2f}s")

    test_izh_step()
    test_grating()
    test_stdp()
    test_single_segment()
    test_multi_segment()
    benchmark()

    if args.extended:
        extended_training()

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  SUMMARY: {n_pass} passed, {n_fail} failed")
    print("=" * 60)

    sys.exit(0 if n_fail == 0 else 1)

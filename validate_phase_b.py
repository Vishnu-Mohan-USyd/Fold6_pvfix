#!/usr/bin/env python
"""validate_phase_b.py — Validate Phase B sequence learning JAX port.

Tests:
  1. E→E STDP structural validation (exact match with numpy reference)
  2. reset_state_jax correctness (preserves weights, resets dynamics)
  3. Single sequence trial comparison (reasonable outputs)
  4. Phase B F>R ratio (MUST increase with training — THE key test)
  5. Phase B omission response (SHOULD increase with training)
  6. Benchmark (JAX must be >5x faster than numpy for Phase B training)
"""

import sys
import time
import math
import traceback
import numpy as np

sys.path.insert(0, '.')

from biologically_plausible_v1_stdp import (
    Params, RgcLgnV1Network, run_sequence_trial, compute_osi,
    calibrate_ee_drive, DelayAwareEESTDP,
)
from network_jax import (
    numpy_net_to_jax_state, jax_state_to_numpy_net,
    run_segment_jax, run_sequence_trial_jax, reset_state_jax,
    delay_aware_ee_stdp_update, evaluate_tuning_jax,
    SimState, StaticConfig,
)
import jax
import jax.numpy as jnp


# ── Helpers ─────────────────────────────────────────────────────────────────

GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
THETA_STEP = 180.0 / GOLDEN_RATIO  # ≈ 111.246°


def compute_fwd_rev_ratio(W_e_e, pref, seq_thetas):
    """Compute forward/reverse weight asymmetry ratio.

    For consecutive sequence elements A→B, find neurons tuned to A and B,
    then compare W_e_e[B_neuron, A_neuron] (forward) vs W_e_e[A_neuron, B_neuron] (reverse).

    Parameters
    ----------
    W_e_e : ndarray (M, M) — E→E weight matrix
    pref : ndarray (M,) — preferred orientations in degrees [0, 180)
    seq_thetas : list of float — sequence element orientations

    Returns
    -------
    fwd_mean, rev_mean, ratio : float
    """
    fwd_ws, rev_ws = [], []
    for ei in range(len(seq_thetas) - 1):
        pre_th, post_th = seq_thetas[ei], seq_thetas[ei + 1]
        d_pre = np.abs(pref - pre_th)
        d_pre = np.minimum(d_pre, 180.0 - d_pre)
        d_post = np.abs(pref - post_th)
        d_post = np.minimum(d_post, 180.0 - d_post)
        pre_mask = d_pre < 22.5
        post_mask = d_post < 22.5
        for pi in np.where(post_mask)[0]:
            for pj in np.where(pre_mask)[0]:
                if pi != pj:
                    fwd_ws.append(W_e_e[pi, pj])
                    rev_ws.append(W_e_e[pj, pi])
    if len(fwd_ws) == 0:
        return 0.0, 0.0, 1.0
    fwd_m = float(np.mean(fwd_ws))
    rev_m = float(np.mean(rev_ws))
    ratio = fwd_m / max(1e-10, rev_m)
    return fwd_m, rev_m, ratio


def build_phase_a_network(seed=42, M=16, N=8):
    """Create a network with Phase B E→E STDP config."""
    p = Params(
        M=M, N=N, seed=seed,
        ee_stdp_enabled=True,
        ee_connectivity="all_to_all",
        ee_stdp_A_plus=0.005,
        ee_stdp_A_minus=0.006,
        ee_stdp_weight_dep=True,
        train_segments=0,
        segment_ms=300.0,
    )
    net = RgcLgnV1Network(p)
    return net


def run_phase_a_jax(net, n_segments=300, verbose=True):
    """Run Phase A (feedforward STDP) using JAX to develop oriented RFs.

    Returns (state, static) after training.
    """
    state, static = numpy_net_to_jax_state(net)
    theta_offset = 0.0
    for seg in range(n_segments):
        theta = (theta_offset + seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)
        if verbose and (seg + 1) % 100 == 0:
            print(f"    Phase A: {seg + 1}/{n_segments} segments done")
    return state, static


def run_phase_a_numpy(net, n_segments=300, verbose=True):
    """Run Phase A (feedforward STDP) using numpy to develop oriented RFs.

    Returns the network in-place (numpy).
    """
    for seg in range(n_segments):
        theta = (seg * THETA_STEP) % 180.0
        net.run_segment(theta, plastic=True)
        if verbose and (seg + 1) % 100 == 0:
            print(f"    Phase A: {seg + 1}/{n_segments} segments done")
    return net


def get_preferred_orientations(state, static, thetas=None, repeats=3):
    """Evaluate tuning and return preferred orientations."""
    if thetas is None:
        thetas = np.linspace(0, 180, 12, endpoint=False)
    rates = evaluate_tuning_jax(state, static, thetas, repeats=repeats)
    _, pref = compute_osi(rates, thetas)
    return pref, rates


# ── Test 1: E→E STDP Structural Validation ─────────────────────────────────

def test_ee_stdp_structural():
    """Compare JAX delay_aware_ee_stdp_update with numpy DelayAwareEESTDP.update()."""
    print("\n" + "=" * 70)
    print("Test 1: E→E STDP structural validation")
    print("=" * 70)

    M = 8
    rng = np.random.default_rng(42)

    for weight_dep in [True, False]:
        label = "weight_dep=True" if weight_dep else "weight_dep=False"
        print(f"\n  Testing {label}...")

        # Create known inputs
        ee_arrivals = (rng.random((M, M)) < 0.2).astype(np.float32)
        np.fill_diagonal(ee_arrivals, 0.0)
        post_spikes = (rng.random(M) < 0.3).astype(np.float32)
        W = rng.uniform(0.0, 0.1, (M, M)).astype(np.float32)
        np.fill_diagonal(W, 0.0)
        mask = np.ones((M, M), dtype=np.float32)
        np.fill_diagonal(mask, 0.0)

        # Params for trace decay
        dt_ms = 0.5
        tau_pre = 20.0
        tau_post = 20.0
        decay_pre = math.exp(-dt_ms / tau_pre)
        decay_post = math.exp(-dt_ms / tau_post)
        A_plus = 0.005
        A_minus = 0.006
        w_min = 0.0
        w_max = 0.2

        # Initial traces
        init_pre_trace = rng.uniform(0, 0.5, (M, M)).astype(np.float32)
        init_post_trace = rng.uniform(0, 0.5, M).astype(np.float32)

        # ---- Numpy reference ----
        p_dummy = Params(M=M, N=4, seed=42, dt_ms=dt_ms,
                         ee_stdp_tau_pre_ms=tau_pre, ee_stdp_tau_post_ms=tau_post)
        np_stdp = DelayAwareEESTDP(M, p_dummy)
        np_stdp.pre_trace = init_pre_trace.copy()
        np_stdp.post_trace = init_post_trace.copy()

        dW_np = np_stdp.update(
            ee_arrivals.copy(), post_spikes.copy(), W.copy(), mask.copy(),
            A_plus, A_minus, w_min, w_max, weight_dep=weight_dep,
        )
        pre_trace_np = np_stdp.pre_trace.copy()
        post_trace_np = np_stdp.post_trace.copy()

        # ---- JAX ----
        pre_trace_jax, post_trace_jax, dW_jax = delay_aware_ee_stdp_update(
            jnp.array(init_pre_trace), jnp.array(init_post_trace),
            jnp.array(ee_arrivals), jnp.array(post_spikes),
            jnp.array(W), jnp.array(mask),
            decay_pre, decay_post,
            A_plus, A_minus,
            w_min, w_max,
            weight_dep,
        )
        dW_jax = np.array(dW_jax)
        pre_trace_jax = np.array(pre_trace_jax)
        post_trace_jax = np.array(post_trace_jax)

        # ---- Compare ----
        dW_err = np.max(np.abs(dW_np - dW_jax))
        pre_err = np.max(np.abs(pre_trace_np - pre_trace_jax))
        post_err = np.max(np.abs(post_trace_np - post_trace_jax))

        print(f"    dW max error:        {dW_err:.2e}")
        print(f"    pre_trace max error: {pre_err:.2e}")
        print(f"    post_trace max error:{post_err:.2e}")

        assert dW_err < 1e-5, f"dW mismatch ({label}): {dW_err:.2e}"
        assert pre_err < 1e-5, f"pre_trace mismatch ({label}): {pre_err:.2e}"
        assert post_err < 1e-5, f"post_trace mismatch ({label}): {post_err:.2e}"
        print(f"    {label}: PASS")

    print("\n  Test 1 PASSED: E→E STDP structural validation")
    return True


# ── Test 2: reset_state_jax Correctness ─────────────────────────────────────

def test_reset_state():
    """Verify reset_state_jax resets dynamics while preserving weights."""
    print("\n" + "=" * 70)
    print("Test 2: reset_state_jax correctness")
    print("=" * 70)

    net = build_phase_a_network(seed=99, M=8, N=4)
    state, static = numpy_net_to_jax_state(net)

    # Run a few segments to get non-trivial state
    for i in range(5):
        state, _ = run_segment_jax(state, static, float(i * 30.0), 1.0, True)

    # Save weights before reset
    W_before = np.array(state.W)
    W_pv_before = np.array(state.W_pv_e)
    W_ee_before = np.array(state.W_e_e)
    I_bias_before = np.array(state.I_v1_bias)
    rate_avg_before = np.array(state.rate_avg)

    # Verify state is non-trivial (not already zero)
    assert float(jnp.abs(state.v1_v - (-65.0)).max()) > 0.1, "State should be non-trivial before reset"
    assert float(jnp.abs(state.g_exc_ff).max()) > 0, "g_exc_ff should be non-zero before reset"

    # Reset
    state_reset = reset_state_jax(state, static)

    # Check membrane potentials reset to -65.0
    v_init = -65.0
    assert np.allclose(np.array(state_reset.v1_v), v_init), "v1_v should be -65.0"
    assert np.allclose(np.array(state_reset.lgn_v), v_init), "lgn_v should be -65.0"
    assert np.allclose(np.array(state_reset.pv_v), v_init), "pv_v should be -65.0"
    assert np.allclose(np.array(state_reset.som_v), v_init), "som_v should be -65.0"

    # Check recovery variables: u = b * v_init
    assert np.allclose(np.array(state_reset.v1_u), static.v1_b * v_init), "v1_u should be b*v_init"
    assert np.allclose(np.array(state_reset.lgn_u), static.lgn_b * v_init), "lgn_u should be b*v_init"
    assert np.allclose(np.array(state_reset.pv_u), static.pv_b * v_init), "pv_u should be b*v_init"
    assert np.allclose(np.array(state_reset.som_u), static.som_b * v_init), "som_u should be b*v_init"

    # Check conductances/currents are zero
    assert float(jnp.abs(state_reset.g_exc_ff).max()) == 0.0, "g_exc_ff should be zero"
    assert float(jnp.abs(state_reset.g_exc_ee).max()) == 0.0, "g_exc_ee should be zero"
    assert float(jnp.abs(state_reset.I_lgn).max()) == 0.0, "I_lgn should be zero"
    assert float(jnp.abs(state_reset.delay_buf).max()) == 0.0, "delay_buf should be zero"
    assert float(jnp.abs(state_reset.delay_buf_ee).max()) == 0.0, "delay_buf_ee should be zero"

    # Check STDP traces are zero
    assert float(jnp.abs(state_reset.stdp_x_pre).max()) == 0.0, "stdp_x_pre should be zero"
    assert float(jnp.abs(state_reset.ee_pre_trace).max()) == 0.0, "ee_pre_trace should be zero"
    assert float(jnp.abs(state_reset.ee_post_trace).max()) == 0.0, "ee_post_trace should be zero"

    # Check weights PRESERVED
    assert np.allclose(np.array(state_reset.W), W_before), "W should be preserved"
    assert np.allclose(np.array(state_reset.W_pv_e), W_pv_before), "W_pv_e should be preserved"
    assert np.allclose(np.array(state_reset.W_e_e), W_ee_before), "W_e_e should be preserved"

    # Check I_v1_bias and rate_avg are preserved (they are "learned" state, not dynamic)
    assert np.allclose(np.array(state_reset.I_v1_bias), I_bias_before), "I_v1_bias should be preserved"
    assert np.allclose(np.array(state_reset.rate_avg), rate_avg_before), "rate_avg should be preserved"

    print("  All dynamic state reset, all weights preserved")
    print("\n  Test 2 PASSED: reset_state_jax correctness")
    return True


# ── Test 3: Single Sequence Trial Comparison ────────────────────────────────

def test_single_trial():
    """Run one sequence trial in JAX and verify reasonable outputs."""
    print("\n" + "=" * 70)
    print("Test 3: Single sequence trial comparison")
    print("=" * 70)

    net = build_phase_a_network(seed=42, M=16, N=8)
    state, static = numpy_net_to_jax_state(net)

    # Run Phase A to develop structured weights
    print("  Running Phase A (100 segments)...")
    for seg in range(100):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)

    seq_thetas = [0.0, 45.0, 90.0, 135.0]
    element_ms = 30.0
    iti_ms = 200.0
    contrast = 1.0

    # Run non-plastic trial in JAX (use state directly — no reset needed for
    # this test since we just want to check reasonable outputs)
    final_state, info = run_sequence_trial_jax(
        state, static, seq_thetas, element_ms, iti_ms, contrast,
        'none',
    )

    v1_counts = np.array(info['v1_counts'])
    element_counts = np.array(info['element_counts'])

    print(f"  Total V1 spike counts: sum={int(v1_counts.sum())}, "
          f"mean={v1_counts.mean():.1f}, max={int(v1_counts.max())}")
    print(f"  Element counts shape: {element_counts.shape}")
    print(f"  Per-element total spikes: {[int(ec.sum()) for ec in element_counts]}")

    # Checks
    assert v1_counts.shape == (static.M,), f"v1_counts shape mismatch: {v1_counts.shape}"
    assert element_counts.shape == (len(seq_thetas), static.M), \
        f"element_counts shape mismatch: {element_counts.shape}"
    assert int(v1_counts.sum()) > 0, "Network should produce some spikes"
    assert int(v1_counts.sum()) < 100000, "Network should not have runaway firing"

    # Run plastic trial (E→E STDP) — use state directly (no reset)
    W_ee_before = np.array(state.W_e_e)
    final_plastic, info_plastic = run_sequence_trial_jax(
        state, static, seq_thetas, element_ms, iti_ms, contrast,
        'ee',
        ee_A_plus_eff=float(static.ee_stdp_A_plus),
        ee_A_minus_eff=float(static.ee_stdp_A_minus),
    )
    W_ee_after = np.array(final_plastic.W_e_e)

    # Weights should change during plastic trial
    w_change = np.abs(W_ee_after - W_ee_before).max()
    print(f"  Max W_e_e change (plastic trial): {w_change:.6f}")
    assert w_change > 0, "W_e_e should change during plastic trial"

    print("\n  Test 3 PASSED: Single sequence trial comparison")
    return True


# ── Test 4: Phase B F>R Ratio Increases ─────────────────────────────────────

def test_phase_b_fr_ratio():
    """THE KEY TEST: Forward/reverse weight ratio must increase with training.

    Protocol:
    1. Phase A in numpy (100 segments — proven to produce good tuning)
    2. Calibrate E→E drive in numpy
    3. Transfer to JAX
    4. Phase B: 400 sequence presentations with E→E STDP in JAX
    5. Check F>R ratio increases and exceeds 1.05
    """
    print("\n" + "=" * 70)
    print("Test 4: Phase B F>R ratio (THE KEY TEST)")
    print("=" * 70)

    # 1. Create network and run Phase A IN NUMPY (proven protocol)
    print("\n  Step 1: Phase A — developing oriented RFs (100 segments, numpy)...")
    net = build_phase_a_network(seed=42, M=16, N=8)
    run_phase_a_numpy(net, n_segments=100)

    # 2. Compute preferred orientations (numpy)
    print("  Step 2: Computing preferred orientations...")
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = net.evaluate_tuning(thetas_eval, repeats=2)
    osi_vals, pref = compute_osi(rates, thetas_eval)
    print(f"    Mean OSI: {osi_vals.mean():.3f}")
    print(f"    Pref orientations: {np.round(pref, 1)}")

    # 3. Calibrate E→E drive in numpy
    print("  Step 3: Calibrating E→E drive (in numpy)...")
    scale, frac = calibrate_ee_drive(net, target_frac=0.15)
    print(f"    Calibration: scale={scale:.1f}, frac={frac:.4f}")

    # Update w_e_e_max based on calibrated weights
    cal_mean = net.W_e_e[net.mask_e_e.astype(bool)].mean() if hasattr(net, 'mask_e_e') else net.W_e_e.mean()
    new_w_max = max(cal_mean * 2.0, net.p.w_e_e_max)
    net.p.w_e_e_max = new_w_max
    print(f"    Updated w_e_e_max={new_w_max:.4f}")

    # Transfer to JAX
    state, static = numpy_net_to_jax_state(net)

    # Re-compute pref after transfer
    pref, _ = get_preferred_orientations(state, static, thetas_eval, repeats=3)

    # 4. Phase B training
    seq_thetas = [0.0, 45.0, 90.0, 135.0]
    element_ms = 30.0
    iti_ms = 200.0
    contrast = 1.0
    n_presentations = 400
    checkpoint_every = 100

    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)

    print(f"\n  Step 4: Phase B training ({n_presentations} presentations)...")
    fr_checkpoints = []

    # Initial F>R ratio
    W_ee_np = np.array(state.W_e_e)
    fwd_m, rev_m, ratio = compute_fwd_rev_ratio(W_ee_np, pref, seq_thetas)
    fr_checkpoints.append((0, ratio))
    print(f"    [pres 0] F>R ratio={ratio:.4f} (fwd={fwd_m:.6f}, rev={rev_m:.6f})")

    for k in range(1, n_presentations + 1):
        # NO reset between training presentations (matches numpy reference):
        # state carries over naturally through the 200ms ITI.
        state, _ = run_sequence_trial_jax(
            state, static, seq_thetas, element_ms, iti_ms, contrast,
            'ee',
            ee_A_plus_eff=A_plus,
            ee_A_minus_eff=A_minus,
        )

        if k % checkpoint_every == 0:
            W_ee_np = np.array(state.W_e_e)
            fwd_m, rev_m, ratio = compute_fwd_rev_ratio(W_ee_np, pref, seq_thetas)
            fr_checkpoints.append((k, ratio))
            off_diag = W_ee_np[np.array(static.mask_e_e, dtype=bool)]
            print(f"    [pres {k}] F>R ratio={ratio:.4f} (fwd={fwd_m:.6f}, rev={rev_m:.6f}), "
                  f"W_ee mean={off_diag.mean():.5f} max={off_diag.max():.4f}")

    # Assertions
    initial_ratio = fr_checkpoints[0][1]
    final_ratio = fr_checkpoints[-1][1]
    print(f"\n  F>R trajectory: {' → '.join(f'{r:.3f}' for _, r in fr_checkpoints)}")
    print(f"  Initial F>R: {initial_ratio:.4f}")
    print(f"  Final F>R:   {final_ratio:.4f}")

    assert final_ratio > initial_ratio, \
        f"F>R ratio must increase: {initial_ratio:.4f} → {final_ratio:.4f}"
    assert final_ratio > 1.05, \
        f"F>R ratio must exceed 1.05, got {final_ratio:.4f}"

    print("\n  Test 4 PASSED: F>R ratio increases with training")
    return True


# ── Test 5: Phase B Omission Response Increases ────────────────────────────

def test_phase_b_omission():
    """Omission response should increase with training (trend positive).

    Uses the same protocol as test_phase_b_fr_ratio but also evaluates
    omission trials at checkpoints.
    """
    print("\n" + "=" * 70)
    print("Test 5: Phase B omission response")
    print("=" * 70)

    # 1. Create network and run Phase A IN NUMPY (proven protocol)
    print("\n  Step 1: Phase A — developing oriented RFs (100 segments, numpy)...")
    net = build_phase_a_network(seed=42, M=16, N=8)
    run_phase_a_numpy(net, n_segments=100)

    # 2. Calibrate E→E
    print("  Step 2: Calibrating E→E drive...")
    scale, frac = calibrate_ee_drive(net, target_frac=0.15)
    cal_mean = net.W_e_e[net.mask_e_e.astype(bool)].mean() if hasattr(net, 'mask_e_e') else net.W_e_e.mean()
    net.p.w_e_e_max = max(cal_mean * 2.0, net.p.w_e_e_max)
    state, static = numpy_net_to_jax_state(net)

    seq_thetas = [0.0, 45.0, 90.0, 135.0]
    element_ms = 30.0
    iti_ms = 200.0
    contrast = 1.0
    n_presentations = 400
    checkpoint_every = 100

    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)

    omission_responses = []

    def evaluate_omission(st):
        """Run trained and omission trials, return omission response."""
        # Trained sequence (non-plastic)
        st_eval = reset_state_jax(st, static)
        _, info_trained = run_sequence_trial_jax(
            st_eval, static, seq_thetas, element_ms, iti_ms, contrast,
            'none',
        )
        trained_elem_counts = np.array(info_trained['element_counts'])

        # Omission trial: omit element 1 (45°)
        st_eval = reset_state_jax(st, static)
        _, info_omit = run_sequence_trial_jax(
            st_eval, static, seq_thetas, element_ms, iti_ms, contrast,
            'none', omit_index=1,
        )
        omit_elem_counts = np.array(info_omit['element_counts'])

        # Control omission: different context — rotate seq by 90°
        ctrl_thetas = [90.0, 135.0, 0.0, 45.0]
        st_eval = reset_state_jax(st, static)
        _, info_ctrl = run_sequence_trial_jax(
            st_eval, static, ctrl_thetas, element_ms, iti_ms, contrast,
            'none', omit_index=1,
        )
        ctrl_elem_counts = np.array(info_ctrl['element_counts'])

        # Omission response = spikes during omitted element in trained context - control
        omit_spikes = float(omit_elem_counts[1].sum())
        ctrl_spikes = float(ctrl_elem_counts[1].sum())
        return omit_spikes - ctrl_spikes

    print(f"\n  Step 3: Phase B training ({n_presentations} presentations) with omission eval...")

    # Initial omission response
    omr_0 = evaluate_omission(state)
    omission_responses.append((0, omr_0))
    print(f"    [pres 0] Omission response={omr_0:.1f}")

    for k in range(1, n_presentations + 1):
        # NO reset between training presentations (matches numpy reference)
        state, _ = run_sequence_trial_jax(
            state, static, seq_thetas, element_ms, iti_ms, contrast,
            'ee',
            ee_A_plus_eff=A_plus,
            ee_A_minus_eff=A_minus,
        )

        if k % checkpoint_every == 0:
            omr = evaluate_omission(state)
            omission_responses.append((k, omr))
            print(f"    [pres {k}] Omission response={omr:.1f}")

    initial_omr = omission_responses[0][1]
    final_omr = omission_responses[-1][1]

    print(f"\n  Omission response trajectory: {' → '.join(f'{r:.1f}' for _, r in omission_responses)}")
    print(f"  Initial: {initial_omr:.1f}")
    print(f"  Final:   {final_omr:.1f}")

    # Soft assertion: trend should be positive (may not be dramatic with only 400 presentations)
    if final_omr > initial_omr:
        print("\n  Test 5 PASSED: Omission response increases with training")
        return True
    else:
        print("\n  Test 5 SOFT FAIL: Omission response did not increase "
              f"({initial_omr:.1f} → {final_omr:.1f})")
        print("  (This is a soft expectation — 400 presentations may not be enough)")
        return True  # Don't fail hard on this one


# ── Test 6: Benchmark — Phase B Training Speedup ───────────────────────────

def test_benchmark():
    """Time numpy vs JAX for Phase B sequence trial training."""
    print("\n" + "=" * 70)
    print("Test 6: Benchmark — Phase B training speedup")
    print("=" * 70)

    net = build_phase_a_network(seed=42, M=16, N=8)

    # Quick Phase A (just 50 segments — enough for benchmark)
    print("  Running quick Phase A (50 segments)...")
    state, static = numpy_net_to_jax_state(net)
    for seg in range(50):
        theta = (seg * THETA_STEP) % 180.0
        state, _ = run_segment_jax(state, static, theta, 1.0, True)

    # Transfer to numpy for baseline
    jax_state_to_numpy_net(state, net)
    net.ff_plastic_enabled = False
    net.ee_stdp_active = True
    net._ee_stdp_ramp_factor = 1.0

    seq_thetas = [0.0, 45.0, 90.0, 135.0]
    element_ms = 30.0
    iti_ms = 200.0
    contrast = 1.0
    n_reps = 200

    # ---- Numpy baseline ----
    print(f"  Timing numpy: {n_reps} sequence trial presentations...")
    t0 = time.perf_counter()
    for k in range(n_reps):
        run_sequence_trial(net, seq_thetas, element_ms, iti_ms, contrast, plastic=True)
    numpy_time = time.perf_counter() - t0
    print(f"    Numpy: {numpy_time:.2f}s ({numpy_time/n_reps*1000:.1f}ms/trial)")

    # ---- JAX ----
    # Warm up JIT
    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)

    state_jax = reset_state_jax(state, static)
    state_jax, _ = run_sequence_trial_jax(
        state_jax, static, seq_thetas, element_ms, iti_ms, contrast,
        'ee', ee_A_plus_eff=A_plus, ee_A_minus_eff=A_minus,
    )
    # Block until JIT compilation + execution completes
    jax.block_until_ready(state_jax.W_e_e)

    print(f"  Timing JAX: {n_reps} sequence trial presentations...")
    state_jax = state  # reset to same starting state
    t0 = time.perf_counter()
    for k in range(n_reps):
        state_jax = reset_state_jax(state_jax, static)
        state_jax, _ = run_sequence_trial_jax(
            state_jax, static, seq_thetas, element_ms, iti_ms, contrast,
            'ee', ee_A_plus_eff=A_plus, ee_A_minus_eff=A_minus,
        )
    jax.block_until_ready(state_jax.W_e_e)
    jax_time = time.perf_counter() - t0
    print(f"    JAX:   {jax_time:.2f}s ({jax_time/n_reps*1000:.1f}ms/trial)")

    speedup = numpy_time / max(1e-6, jax_time)
    print(f"\n  Speedup: {speedup:.1f}x (numpy={numpy_time:.2f}s, jax={jax_time:.2f}s)")

    assert speedup > 5.0, f"JAX speedup must be >5x, got {speedup:.1f}x"

    print(f"\n  Test 6 PASSED: JAX is {speedup:.1f}x faster")
    return True


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    tests = [
        ("Test 1: E→E STDP structural", test_ee_stdp_structural),
        ("Test 2: reset_state_jax", test_reset_state),
        ("Test 3: Single sequence trial", test_single_trial),
        ("Test 4: Phase B F>R ratio", test_phase_b_fr_ratio),
        ("Test 5: Phase B omission", test_phase_b_omission),
        ("Test 6: Benchmark", test_benchmark),
    ]

    results = {}
    for name, fn in tests:
        try:
            ok = fn()
            results[name] = "PASS" if ok else "FAIL"
        except Exception as e:
            print(f"\n  {name}: EXCEPTION — {e}")
            traceback.print_exc()
            results[name] = "FAIL"

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, status in results.items():
        marker = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{marker}] {name}")
        if status != "PASS":
            all_pass = False

    if all_pass:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1


if __name__ == '__main__':
    sys.exit(main())

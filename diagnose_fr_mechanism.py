#!/usr/bin/env python
"""diagnose_fr_mechanism.py — Measure spike timing and dW contributions
that explain WHY F>R fails with 300 Phase A segments.

Key measurements:
1. Per-element spike timing distributions (when do neurons fire?)
2. Cross-element spike timing: Δt between last pre-neuron spikes and first
   post-neuron spikes for consecutive elements
3. Decompose dW into within-element vs cross-element contributions
4. Compare 100 vs 300 Phase A segments
"""

import sys
import time
import math
import numpy as np

sys.path.insert(0, '.')

from biologically_plausible_v1_stdp import (
    Params, RgcLgnV1Network, compute_osi, calibrate_ee_drive,
)
from network_jax import (
    numpy_net_to_jax_state, run_segment_jax, run_sequence_trial_jax,
    reset_state_jax, evaluate_tuning_jax,
    delay_aware_ee_stdp_update, build_trial_stimulus,
    timestep, SimState, StaticConfig,
)
import jax
import jax.numpy as jnp

GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
THETA_STEP = 180.0 / GOLDEN_RATIO
SEQ_THETAS = [0.0, 45.0, 90.0, 135.0]
ELEMENT_MS = 150.0
ITI_MS = 1500.0
CONTRAST = 1.0


def compute_fwd_rev_ratio(W_e_e, pref, seq_thetas):
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


def get_neuron_groups(pref, seq_thetas, bw=22.5):
    """Return dict mapping element index to list of neuron indices tuned near that theta."""
    groups = {}
    for ei, th in enumerate(seq_thetas):
        d = np.abs(pref - th)
        d = np.minimum(d, 180.0 - d)
        groups[ei] = np.where(d < bw)[0]
    return groups


def setup_condition(phase_a_seg, seed=42):
    """Run Phase A + calibration and return (state, static, pref, net)."""
    p = Params(
        M=16, N=8, seed=seed,
        ee_stdp_enabled=True, ee_connectivity="all_to_all",
        ee_stdp_A_plus=0.005, ee_stdp_A_minus=0.006,
        ee_stdp_weight_dep=True, train_segments=0, segment_ms=300.0,
    )
    net = RgcLgnV1Network(p)

    # Phase A in numpy (deterministic, well-tested)
    print(f"  Phase A ({phase_a_seg} segments, numpy)...")
    t0 = time.perf_counter()
    for seg in range(phase_a_seg):
        theta = (seg * THETA_STEP) % 180.0
        net.run_segment(theta, plastic=True)
        if (seg + 1) % 100 == 0:
            print(f"    {seg+1}/{phase_a_seg}")
    print(f"    Done in {time.perf_counter()-t0:.1f}s")

    # Evaluate tuning
    thetas_eval = np.linspace(0, 180, 12, endpoint=False)
    rates = net.evaluate_tuning(thetas_eval, repeats=2)
    osi_vals, pref = compute_osi(rates, thetas_eval)
    print(f"    OSI: {osi_vals.mean():.3f}")

    # Calibrate E→E
    scale, frac = calibrate_ee_drive(net, target_frac=0.15)
    mask_ee = net.mask_e_e.astype(bool)
    cal_mean = float(net.W_e_e[mask_ee].mean())
    new_w_max = max(cal_mean * 3.0, net.p.w_e_e_max)
    net.p.w_e_e_max = new_w_max
    print(f"    Calibration: scale={scale:.1f}, cal_mean={cal_mean:.4f}, w_max={new_w_max:.4f}")

    state, static = numpy_net_to_jax_state(net)
    # Re-evaluate pref with calibrated state
    rates2 = evaluate_tuning_jax(state, static, thetas_eval, repeats=3)
    _, pref = compute_osi(rates2, thetas_eval)

    return state, static, pref, net


def measure_spike_timing_per_trial(state, static, seq_thetas, element_ms, iti_ms,
                                    contrast, pref, n_trials=5):
    """Run non-plastic trials and collect detailed spike timing info.

    Returns:
      per_element_rates: (n_trials, n_elem, M) spike counts
      spike_rasters: list of (n_trials,) each with per-timestep spike arrays
    """
    n_elem = len(seq_thetas)
    element_steps = int(round(element_ms / static.dt_ms))
    iti_steps = int(round(iti_ms / static.dt_ms))
    total_steps = n_elem * element_steps + iti_steps
    M = int(static.M)

    groups = get_neuron_groups(pref, seq_thetas)

    all_spike_times = []  # per trial: dict of element_idx -> array of spike times for tuned neurons

    for trial in range(n_trials):
        st = reset_state_jax(state, static)

        # Build stimulus
        key = st.rng_key
        key, *phase_keys = jax.random.split(key, n_elem + 1)
        phases = jnp.array([jax.random.uniform(pk, (), minval=0.0, maxval=2*jnp.pi) for pk in phase_keys])
        key, subkey = jax.random.split(key)
        step_keys = jax.random.split(subkey, total_steps)

        theta_arr, contrast_arr, phase_arr, t_ms_arr = build_trial_stimulus(
            seq_thetas, n_elem, element_steps, iti_steps, contrast, phases, -1, static.dt_ms)

        # Run non-plastic and collect per-timestep spikes
        # We need per-step spike data, so run step-by-step in a scan
        # Use the non-plastic runner but collect spike raster
        st_run, info = run_sequence_trial_jax(
            st._replace(rng_key=key), static, seq_thetas, element_ms, iti_ms, contrast,
            'none', step_keys=step_keys, phases=phases)

        # element_counts shape: (n_elem, M)
        elem_counts = np.array(info['element_counts'])

        # For timing within elements, we need finer data
        # Use g_exc_ee_traces as a proxy for activity timing
        # shape: (n_elem, element_steps, M)
        g_traces = np.array(info['g_exc_ee_traces'])

        trial_data = {
            'elem_counts': elem_counts,
            'g_traces': g_traces,
        }

        # Compute per-group mean firing rates
        for ei in range(n_elem):
            nids = groups.get(ei, np.array([], dtype=int))
            if len(nids) > 0:
                trial_data[f'group_{ei}_count'] = float(elem_counts[ei, nids].mean())
                # When does g_exc_ee peak for this group? (proxy for spike timing)
                g_group = g_traces[ei, :, nids].mean(axis=1)  # (element_steps,)
                if g_group.max() > 0:
                    peak_t = np.argmax(g_group) * float(static.dt_ms)
                    trial_data[f'group_{ei}_peak_ms'] = peak_t
                else:
                    trial_data[f'group_{ei}_peak_ms'] = -1.0

        all_spike_times.append(trial_data)

    return all_spike_times, groups


def measure_dw_decomposition(state, static, pref, n_presentations=10):
    """Run n plastic presentations and measure cumulative dW for fwd vs rev pairs.

    Returns dw_fwd_cumulative, dw_rev_cumulative per presentation.
    """
    groups = get_neuron_groups(pref, SEQ_THETAS)
    M = int(static.M)

    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)

    fwd_pairs = []  # (post, pre) indices for forward direction
    rev_pairs = []  # (post, pre) indices for reverse direction

    for ei in range(len(SEQ_THETAS) - 1):
        pre_neurons = groups.get(ei, np.array([], dtype=int))
        post_neurons = groups.get(ei + 1, np.array([], dtype=int))
        for pi in post_neurons:
            for pj in pre_neurons:
                if pi != pj:
                    fwd_pairs.append((pi, pj))
                    rev_pairs.append((pj, pi))

    print(f"    Forward pairs: {len(fwd_pairs)}, Reverse pairs: {len(rev_pairs)}")
    if len(fwd_pairs) == 0:
        print("    WARNING: No fwd/rev pairs found!")
        return [], []

    dw_fwd_history = []
    dw_rev_history = []

    st = state
    for k in range(n_presentations):
        W_before = np.array(st.W_e_e)

        st, _ = run_sequence_trial_jax(
            st, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST,
            'ee', ee_A_plus_eff=A_plus, ee_A_minus_eff=A_minus)

        W_after = np.array(st.W_e_e)
        dW = W_after - W_before

        # Extract dW for forward and reverse pairs
        fwd_dws = [dW[pi, pj] for pi, pj in fwd_pairs]
        rev_dws = [dW[pi, pj] for pi, pj in rev_pairs]

        dw_fwd_history.append(np.mean(fwd_dws))
        dw_rev_history.append(np.mean(rev_dws))

        if (k + 1) % 5 == 0 or k == 0:
            fwd_m, rev_m, ratio = compute_fwd_rev_ratio(W_after, pref, SEQ_THETAS)
            print(f"    [pres {k+1:3d}] dW_fwd={np.mean(fwd_dws):+.6f}, "
                  f"dW_rev={np.mean(rev_dws):+.6f}, "
                  f"F>R={ratio:.4f}")

    return dw_fwd_history, dw_rev_history


def analyze_trace_dynamics(static):
    """Compute theoretical trace decay over key time intervals."""
    dt = float(static.dt_ms)
    tau_pre = float(static.ee_stdp_tau_pre_ms) if hasattr(static, 'ee_stdp_tau_pre_ms') else 20.0
    tau_post = float(static.ee_stdp_tau_post_ms) if hasattr(static, 'ee_stdp_tau_post_ms') else 20.0

    # Get tau values from the decay constants instead
    decay_pre = float(static.ee_stdp_decay_pre)
    decay_post = float(static.ee_stdp_decay_post)
    # decay = exp(-dt/tau) => tau = -dt / ln(decay)
    tau_pre_eff = -dt / np.log(decay_pre) if decay_pre > 0 else float('inf')
    tau_post_eff = -dt / np.log(decay_post) if decay_post > 0 else float('inf')

    print(f"\n  STDP Trace Dynamics:")
    print(f"    dt = {dt} ms")
    print(f"    tau_pre  = {tau_pre_eff:.1f} ms (decay_pre = {decay_pre:.6f})")
    print(f"    tau_post = {tau_post_eff:.1f} ms (decay_post = {decay_post:.6f})")
    print(f"    A_plus   = {float(static.ee_stdp_A_plus)}")
    print(f"    A_minus  = {float(static.ee_stdp_A_minus)}")
    print(f"    w_e_e_max = {float(static.w_e_e_max)}")
    print(f"    w_e_e_min = {float(static.w_e_e_min)}")
    print(f"    weight_dep = {bool(static.ee_stdp_weight_dep)}")

    # Trace remaining after key intervals
    intervals = [1, 5, 10, 20, 50, 100, 150, 200, 300]
    print(f"\n    Trace remaining after interval (fraction of initial value):")
    print(f"    {'Interval (ms)':>15} {'pre_trace':>12} {'post_trace':>12}")
    for dt_interval in intervals:
        n_steps = int(dt_interval / dt)
        pre_remain = decay_pre ** n_steps
        post_remain = decay_post ** n_steps
        print(f"    {dt_interval:>12d} ms {pre_remain:>12.6f} {post_remain:>12.6f}")

    return tau_pre_eff, tau_post_eff


def analyze_weight_dependent_stdp(static, W_mean):
    """Analyze the weight-dependent STDP equilibrium and effective rates."""
    A_plus = float(static.ee_stdp_A_plus)
    A_minus = float(static.ee_stdp_A_minus)
    w_min = float(static.w_e_e_min)
    w_max = float(static.w_e_e_max)

    print(f"\n  Weight-Dependent STDP Analysis (at W_mean={W_mean:.4f}):")
    eff_ltp = A_plus * (w_max - W_mean)
    eff_ltd = A_minus * (W_mean - w_min)
    print(f"    Effective LTP rate: A+ × (w_max - W) = {A_plus} × {w_max - W_mean:.4f} = {eff_ltp:.6f}")
    print(f"    Effective LTD rate: A- × (W - w_min) = {A_minus} × {W_mean - w_min:.4f} = {eff_ltd:.6f}")
    print(f"    LTP/LTD ratio: {eff_ltp / max(1e-10, eff_ltd):.4f}")

    # Equilibrium weight: A+ × (w_max - w_eq) = A- × (w_eq - w_min)
    # A+ × w_max - A+ × w_eq = A- × w_eq - A- × w_min
    # w_eq × (A+ + A-) = A+ × w_max + A- × w_min
    w_eq = (A_plus * w_max + A_minus * w_min) / (A_plus + A_minus)
    print(f"    Equilibrium weight: w_eq = {w_eq:.4f}")
    print(f"    W_mean vs w_eq: {'ABOVE' if W_mean > w_eq else 'BELOW'} equilibrium "
          f"(Δ = {W_mean - w_eq:+.4f})")

    # At equilibrium, dW_ltp = dW_ltd for equal pre-post correlations.
    # For F>R to emerge, forward pairs need net LTP and reverse pairs need net LTD.
    # With weight-dep STDP, if W > w_eq, LTD dominates → ALL weights drift down.
    # This is a problem if calibration sets W > w_eq.
    if W_mean > w_eq:
        print(f"    *** CRITICAL: W_mean ({W_mean:.4f}) > w_eq ({w_eq:.4f})")
        print(f"        → LTD dominates for ALL synapses regardless of direction")
        print(f"        → Even forward synapses lose weight → F>R CANNOT increase")

    return w_eq


def main():
    print("=" * 80)
    print("Diagnostic: F>R mechanism analysis")
    print("=" * 80)

    for phase_a_seg in [100, 300]:
        print(f"\n{'='*80}")
        print(f"CONDITION: {phase_a_seg} Phase A segments")
        print(f"{'='*80}")

        state, static, pref, net = setup_condition(phase_a_seg)
        groups = get_neuron_groups(pref, SEQ_THETAS)

        # Print neuron coverage
        print(f"\n  Neuron groups (22.5° bandwidth):")
        for ei, th in enumerate(SEQ_THETAS):
            nids = groups[ei]
            print(f"    Element {ei} (θ={th}°): {len(nids)} neurons — {nids}")

        # Print calibrated weight stats
        W_ee = np.array(state.W_e_e)
        mask_ee = np.array(static.mask_e_e, dtype=bool)
        off_diag = W_ee[mask_ee]
        print(f"\n  Calibrated E→E weights:")
        print(f"    mean={off_diag.mean():.4f}, std={off_diag.std():.4f}, "
              f"max={off_diag.max():.4f}, min={off_diag.min():.4f}")

        # Trace dynamics
        tau_pre, tau_post = analyze_trace_dynamics(static)

        # Weight-dependent equilibrium
        w_eq = analyze_weight_dependent_stdp(static, off_diag.mean())

        # Spike timing per trial
        print(f"\n  Measuring spike timing (5 non-plastic trials)...")
        trial_data, groups = measure_spike_timing_per_trial(
            state, static, SEQ_THETAS, ELEMENT_MS, ITI_MS, CONTRAST, pref, n_trials=5)

        for trial in range(min(3, len(trial_data))):
            td = trial_data[trial]
            print(f"\n    Trial {trial+1}:")
            for ei in range(len(SEQ_THETAS)):
                nids = groups.get(ei, np.array([]))
                count = td.get(f'group_{ei}_count', 0)
                peak = td.get(f'group_{ei}_peak_ms', -1)
                total = td['elem_counts'][ei].sum()
                print(f"      Element {ei} (θ={SEQ_THETAS[ei]}°): "
                      f"group_rate={count:.1f}, total_spikes={total}, "
                      f"g_exc_ee peak={peak:.0f}ms")

        # Cross-element timing analysis
        print(f"\n  Cross-element timing (g_exc_ee traces):")
        for trial in range(min(2, len(trial_data))):
            td = trial_data[trial]
            g = td['g_traces']  # (n_elem, element_steps, M)
            for ei in range(len(SEQ_THETAS) - 1):
                pre_nids = groups.get(ei, np.array([]))
                post_nids = groups.get(ei + 1, np.array([]))
                if len(pre_nids) > 0 and len(post_nids) > 0:
                    # g_exc_ee for pre-neurons during current element
                    g_pre_during = g[ei, :, pre_nids].mean(axis=1)
                    # g_exc_ee for post-neurons during next element
                    g_post_during = g[ei + 1, :, post_nids].mean(axis=1)
                    # When does pre-neuron's recurrent activity peak?
                    pre_peak_t = np.argmax(g_pre_during)
                    # When does post-neuron's recurrent activity start?
                    post_onset_t = np.argmax(g_post_during > g_post_during.max() * 0.1)

                    # The critical gap: time from end of element ei to start of element ei+1
                    # In continuous time: end of ei at element_steps, start of ei+1 at 0
                    # Elements are back-to-back, so gap = 0
                    # But the trace gap matters: pre traces from end of elem ei decay
                    # during all of elem ei+1
                    print(f"    Trial {trial+1}, {ei}→{ei+1}: "
                          f"pre g_ee peak at {pre_peak_t}ms, "
                          f"post g_ee onset at {post_onset_t}ms, "
                          f"pre g_ee at end: {g_pre_during[-1]:.6f}, "
                          f"post g_ee at start: {g_post_during[0]:.6f}")

        # dW decomposition
        print(f"\n  Measuring dW decomposition (20 plastic presentations)...")
        dw_fwd, dw_rev = measure_dw_decomposition(state, static, pref, n_presentations=20)

        if len(dw_fwd) > 0:
            print(f"\n  dW Summary:")
            print(f"    Mean dW_fwd per presentation: {np.mean(dw_fwd):+.6f}")
            print(f"    Mean dW_rev per presentation: {np.mean(dw_rev):+.6f}")
            print(f"    dW_fwd - dW_rev (asymmetry): {np.mean(dw_fwd) - np.mean(dw_rev):+.6f}")
            print(f"    Ratio of mean dW: fwd/rev = "
                  f"{np.mean(dw_fwd) / max(1e-10, np.mean(dw_rev)):.4f}")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

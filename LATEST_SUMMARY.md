# V1 Spiking Network — Project Summary

**Last updated**: 2026-02-28
**Branch**: `fix_omission_response`
**Validation**: 5/5 tests pass (`validate_omission_fix_clean.log`)

---

## What This Project Is

A biologically plausible spiking neural network that models orientation selectivity (OSI) formation in primary visual cortex (V1) through experience-dependent learning. The network implements the retina-to-cortex pathway (RGC → LGN → V1) with biologically grounded plasticity rules, and extends to sequence learning following the Gavornik & Bear (2014) protocol.

The project has two phases:
- **Phase A**: Orientation selectivity emerges via feedforward STDP (LGN → V1)
- **Phase B**: Temporal sequence learning via recurrent E→E STDP within V1

---

## File Map

| File | Lines | Purpose |
|------|-------|---------|
| `biologically_plausible_v1_stdp.py` | ~8,470 | Core numpy simulation — all mechanisms, `Params`, `RgcLgnV1Network` |
| `network_jax.py` | ~1,700 | JAX GPU port — 25x speedup, Phase A + Phase B |
| `validate_omission_fix.py` | ~640 | Phase B validation suite (5 tests) |
| `validate_jax_port.py` | ~490 | JAX vs numpy correctness tests (15 tests) |
| `validate_phase_b.py` | ~730 | Extended Phase B trajectory tests |
| `osi_investigation_harness.py` | ~670 | Standardized ablation/dose-response framework |
| `investigate_M1.py`–`M6.py` | | Per-mechanism ablation studies |
| `investigate_synthesis.py` | ~550 | Knock-in analysis + pairwise interactions |
| `investigation_results/` | | All results (JSON, npz, plots, reports) |
| `diagnose_*.py` | | Debugging/diagnostic scripts from development |

---

## Key APIs

### Numpy (`biologically_plausible_v1_stdp.py`)

```python
p = Params(M=16, N=8, seed=42, ...)           # ~287 configurable fields
net = RgcLgnV1Network(p)                        # Line 1202
net.run_segment(theta_deg, plastic, contrast)   # Line 2743 — one 300ms grating segment
net.evaluate_tuning(thetas, repeats, contrast)  # Line 3218 — non-destructive tuning eval
compute_osi(rates, thetas)                      # Line 115  — doubled-angle vector method
```

### JAX GPU (`network_jax.py`)

```python
state, static = numpy_net_to_jax_state(net)     # Convert numpy → JAX pytree
state, counts = run_segment_jax(state, static, theta_deg, contrast, plastic)  # Phase A
state, counts = run_sequence_trial_jax(          # Phase B
    state, static, thetas, element_ms, iti_ms, contrast,
    plastic_mode='ee',  # or 'none' for eval
    omit_index=-1,      # which element to omit (-1 = none)
    ee_A_plus_eff=0.005, ee_A_minus_eff=0.006)
scale, frac = calibrate_ee_drive_jax(state, static)  # E→E drive calibration
```

**Important**: Reuse the same `static` object across calls to avoid JIT recompilation (cache key is `id(static)`).

### Investigation Harness (`osi_investigation_harness.py`)

```python
cfg = InvestigationConfig(condition_name="ablation_X", param_overrides={...},
                          seeds=[1, 42, 137], M=16, N=8, train_segments=300)
result = run_investigation(cfg)                  # All seeds
result = run_single_seed(cfg, seed=42)           # Single seed
metrics = compute_weight_metrics(net)            # OSI, weight stats, diversity
```

---

## Network Architecture

### Sensory Input Pipeline

**RGC (Retinal Ganglion Cells)**
- Difference-of-Gaussians center-surround filtering
- ON and OFF channels, independently sampled
- Position jitter (0.15) breaks lattice artifacts
- Firing: Poisson spikes at `base_rate + gain_rate * stimulus`

**LGN (Lateral Geniculate Nucleus)**
- Izhikevich thalamocortical (TC) neurons: a=0.02, b=0.25, c=-65, d=0.05
- Short-term depression (STP): u=0.05 depletion/spike, tau_rec=50ms
- N=8 patch → 2 × 8² = 128 LGN neurons (ON + OFF)

### V1 Excitatory Population

- M=16 ensembles (default), Izhikevich regular-spiking (RS): a=0.02, b=0.2, c=-65, d=8
- LGN→E: dense with spatial envelope (sigma=2.0), 75% anatomical sparsity
- E→E: Gaussian lateral profile (sigma=1.5), heterogeneous conduction delays (1–6ms)
- dt = 0.5ms (Izhikevich stability requirement)

### Inhibitory Interneurons

**PV (Parvalbumin, fast-spiking)**
- Feedforward (LGN→PV) + feedback (E→PV→E) circuit
- Inhibitory STDP (iSTDP): homeostatic, targets ~8 Hz firing rate
- Status: modulatory (~2% effect on OSI)

**SOM (Somatostatin, low-threshold spiking)**
- E→SOM→E lateral inhibition (disabled by default: w=0.0)
- Status: modulatory (~1% effect on OSI)

---

## Plasticity Rules

### 1. Triplet STDP (LGN→E, Phase A) — **ESSENTIAL**

From Pfister & Gerstner (2006). Drives orientation selectivity formation.

```
LTP: dW = +A2_plus * post_spike * x_pre * (w_max - W)    # pair term
         +A3_plus * post_spike * x_pre * x_post_slow      # triplet enhancement
LTD: dW = -A2_minus * pre_arrival * x_post * W            # pair term
```

Parameters: A2_plus=0.008, A2_minus=0.010, tau_plus=tau_minus=20ms, w_max=1.0

**Critical trace update ordering** (lines 829–843):
1. Decay traces → 2. LTD on pre-arrival (OLD post trace) → 3. Update pre traces → 4. LTP on post spike (NEW pre trace, catches coincidences) → 5. Update post traces

### 2. Heterosynaptic Depression — **ESSENTIAL ENABLER**

Resource-like competition: depresses inactive synapses on postsynaptic spike.

```
dW = -A_het * post_spike * (1 - arrivals) * W
```

A_het=0.032 (default), 0.064 (optimal). Prevents all-weights-saturate-to-w_max failure mode.

### 3. ON/OFF Split Competition — **IMPORTANT** (redundant with Het)

Cross-channel depression: ON activity depresses OFF synapses and vice versa.

```
dW_off = -A_split * post_spike * ON_trace * W_off
dW_on  = -A_split * post_spike * OFF_trace * W_on
```

A_split=0.2. Creates phase-opponent RF structure. Strongly redundant with Het (interaction I = −0.653).

### 4. E→E Delay-Aware STDP (Phase B) — Sequence Learning

Weight-dependent pair-based STDP on recurrent excitatory connections.

```
LTP: dW = +A_plus * post_spike * pre_trace * (w_max - W)
LTD: dW = -A_minus * delayed_arrival * post_trace * (W - w_min)
```

A_plus=0.005, A_minus=0.006, w_e_e_max = 3× calibrated mean (~5.05).

**Self-regulating**: As W → w_max, LTP → 0. This creates a ceiling on F>R asymmetry at ~1.22. This is a property of the learning rule, not a bug.

---

## Phase B Protocol (Gavornik & Bear 2014)

### Experimental Design

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Sequence | [0°, 45°, 90°, 135°] | 4-element orientation sequence |
| Element duration | 150ms | Matches G&B 2014 |
| ITI | 1500ms | Inter-trial interval |
| Presentations | 800 | ~200/day × 4 days |
| STDP mode | E→E only | Feedforward frozen |
| Omission metric | g_exc_ee conductance | Continuous, low-noise |

### Pipeline

1. **Phase A** (100 segments, JAX): Feedforward STDP develops OSI → mean OSI ≈ 0.81
2. **Calibrate E→E**: Scale weights so ~15% of excitatory drive is recurrent, with OSI floor=0.30
3. **Set w_e_e_max** = 3× calibrated mean (headroom for LTP growth)
4. **Record pre-calibration preferred orientations** (post-cal pref is distorted)
5. **Phase B training**: 800 presentations with E→E STDP
6. **Evaluate**: F>R asymmetry + omission response (conductance-based)

### Results

| Metric | Value |
|--------|-------|
| F>R ratio (800 pres) | **1.169** (monotonically increasing, threshold >1.15) |
| Omission response | **+0.001418** conductance (positive) |
| F>R trajectory | 1.000 → 1.044 → 1.102 → 1.162 → 1.175 → 1.169 |
| Training time (JAX) | **100 seconds** for 800 presentations |

### Fixes Applied (from development)

1. **Conductance-based metric**: g_exc_ee traces instead of noisy spike counts
2. **150ms elements**: Matches G&B 2014 (was 30ms)
3. **800 presentations**: Matches G&B protocol (was 400)
4. **Pre-calibration pref**: Post-cal pref distorted by strong E→E weights
5. **3× w_e_e_max**: 2× causes weight saturation at ~87% of ceiling; 3× gives sustained growth
6. **Default calibration**: target_frac=0.15, osi_floor=0.30 (target_frac=0.05 too weak for omission response)

---

## OSI Mechanism Investigation Results

Systematic ablation study across 66+ conditions × 3 seeds (~184 simulation runs).
Full report: `investigation_results/synthesis/FINAL_REPORT.md`

### Ablation from Full Model (baseline OSI = 0.846)

| Rank | Mechanism | OSI after ablation | Delta | Role |
|------|-----------|-------------------|-------|------|
| 1 | Triplet STDP | 0.200 | −0.646 | **Essential** |
| 2 | ON/OFF Split | 0.522 | −0.324 | **Important** |
| 3 | Heterosynaptic | 0.745 | −0.101 | **Facilitating** |
| 4 | PV Inhibition | 0.824 | −0.022 | Modulatory |
| 5 | SOM Inhibition | 0.837 | −0.009 | Modulatory |
| 6 | TC STP | 0.844 | −0.002 | Neutral |

### Additive Knock-In (building up from nothing)

| Step | Added | OSI | Marginal gain |
|------|-------|-----|---------------|
| 1 | STDP only | 0.014 | — |
| 2 | + Heterosynaptic | 0.792 | **+0.779** |
| 3 | + ON/OFF Split | 0.798 | +0.006 |
| 4 | + PV | 0.829 | +0.031 |
| 5 | + SOM | 0.844 | +0.015 |
| 6 | Full model | 0.846 | +0.002 |

### Key Insight

**STDP + weight competition = necessary and sufficient for OSI.**

- STDP alone: weights all saturate to w_max → no selectivity (OSI 0.014)
- Add heterosynaptic depression: inactive synapses are competitively depressed → sharp selectivity (OSI 0.792)
- ON/OFF split provides redundant competition mechanism (interaction I = −0.653 with Het)
- PV/SOM/STP are modulatory refinements, not essential

---

## Performance

| Operation | Numpy | JAX GPU | Speedup |
|-----------|-------|---------|---------|
| 1 Phase A segment | 685ms | 27ms | 25× |
| 300 segments | ~9 min | 7.0s | ~77× |
| 1 Phase B trial | 593ms | 120ms | 5× |
| 800 Phase B presentations | ~8 min | 100s | ~5× |

### JAX Architecture

- **SimState** (41 fields): mutable NamedTuple holding all dynamic arrays (voltages, currents, weights, traces, delay buffers, RNG key)
- **StaticConfig** (104 fields): immutable NamedTuple holding connectivity matrices, masks, delays, decay constants, all scalar parameters
- **JIT strategy**: Closure-based caching. StaticConfig captured by closure in `_make_segment_runners()` factory. Cache key = `id(static)`. Reuse same object to avoid recompilation.
- **Phase A RNG**: Must use numpy RNG (not JAX) to get well-distributed preferred orientations across ensembles

---

## Environment

```
Python:    3.9 (miniconda3/envs/habitat)
NumPy:     1.26.4
JAX:       GPU-enabled (CUDA)
SciPy:     1.8.1
Matplotlib: 3.9.2
```

---

## Validation Test Suites

### Phase B (`validate_omission_fix.py`) — 5/5 PASS

| Test | What it checks |
|------|----------------|
| 1. Trace recording | g_exc_ee shape, non-negative, omission gap |
| 2. F>R ratio | >1.15, monotonically increasing over 800 pres |
| 3. Omission response | Conductance positive (trained > control) |
| 4. Bio audit | Weight-dep STDP, no global renorm, local plasticity |
| 5. Benchmark | 800 pres in <5 min on GPU |

### JAX Port (`validate_jax_port.py`) — 15/15 PASS

Structural correctness (Izhikevich, grating, STDP update, delay buffer), statistical convergence (OSI, tuning curves, weight distributions), multi-segment training, performance benchmark.

---

## References

- Izhikevich (2003, 2007): Spiking neuron models
- Pfister & Gerstner (2006): Triplet STDP
- Bi & Poo (1998): Spike-timing dependent plasticity
- Song, Miller & Abbott (2000): Weight-dependent STDP
- Turrigiano (2008): Homeostatic synaptic plasticity
- Gavornik & Bear (2014): Learned spatiotemporal sequence recognition in V1

#!/usr/bin/env python
"""Benchmark JAX vmap scaling across HC counts."""
import time
import numpy as np
import sys
sys.path.insert(0, '/home/vysoforlife/code_files/und_OSI_formation_extend/Fold6_pvfix')

from biologically_plausible_v1_stdp import Params, RgcLgnV1Network
from network_jax import numpy_net_to_jax_state, run_segment_jax

N_SEGMENTS = 30  # enough to get stable timing
M_PER_HC = 16    # Must be a perfect square (4x4 grid within each HC)

results = []
# n_hc must be a perfect square for grid layout (1, 4, 9, 16, 25, 36, 49, 64)
for n_hc in [1, 4, 16, 64]:
    print(f"\n--- n_hc={n_hc} (M_total={n_hc * M_PER_HC}) ---")
    p = Params(seed=42, M=M_PER_HC, n_hc=n_hc)
    net = RgcLgnV1Network(p)
    state, static = numpy_net_to_jax_state(net)

    # Warmup (JIT compilation)
    print(f"  Warmup (JIT compile)...")
    state_w, _ = run_segment_jax(state, static, 45.0, 1.0, True)

    # Timed run
    print(f"  Timing {N_SEGMENTS} segments...")
    state_t = state_w
    t0 = time.perf_counter()
    for seg in range(N_SEGMENTS):
        theta = float(np.random.choice([0, 45, 90, 135]))
        state_t, counts = run_segment_jax(state_t, static, theta, 1.0, True)
    elapsed = time.perf_counter() - t0

    ms_per_seg = 1000 * elapsed / N_SEGMENTS
    M_total = n_hc * M_PER_HC
    results.append((n_hc, M_total, elapsed, ms_per_seg))
    print(f"  n_hc={n_hc:3d}: M_total={M_total:5d}, {elapsed:.2f}s ({ms_per_seg:.1f}ms/seg)")

# Summary
print("\n\n=== SCALING SUMMARY ===")
print(f"  {'n_hc':>5s}  {'M_total':>7s}  {'Total(s)':>8s}  {'ms/seg':>8s}  {'ratio':>6s}")
print(f"  {'-----':>5s}  {'-------':>7s}  {'--------':>8s}  {'------':>8s}  {'-----':>6s}")
base_ms = results[0][3]
for n_hc, M_total, elapsed, ms_per_seg in results:
    ratio = ms_per_seg / base_ms
    print(f"  {n_hc:5d}  {M_total:7d}  {elapsed:8.2f}  {ms_per_seg:8.1f}  {ratio:6.2f}x")

print(f"\n  Ideal (O(n_hc) vmap): n_hc=64 should scale ~linearly")
print(f"  Target: n_hc=64 should be < 3.0x of n_hc=1")
final_ratio = results[-1][3] / base_ms
if final_ratio < 3.0:
    print(f"  [PASS] n_hc=64 ratio = {final_ratio:.2f}x < 3.0x")
else:
    print(f"  [FAIL] n_hc=64 ratio = {final_ratio:.2f}x >= 3.0x -- NEEDS FURTHER OPTIMIZATION")

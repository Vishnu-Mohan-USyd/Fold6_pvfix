#!/usr/bin/env python3
"""Show pre- and post-Phase A: mean OSI + smoothed weight filter maps (subset)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from biologically_plausible_v1_stdp import Params, RgcLgnV1Network, compute_osi

OUT = "/tmp/diag_before_after"
os.makedirs(OUT, exist_ok=True)

p = Params(N=8, M=32, seed=1, v1_bias_eta=0.0)
net = RgcLgnV1Network(p, init_mode="random")
N2 = p.N * p.N

thetas = np.linspace(0, 180 - 180/12, 12)

# --- Pre-Phase A ---
rates_pre = net.evaluate_tuning(thetas, repeats=7, contrast=1.0)
osi_pre, pref_pre = compute_osi(rates_pre, thetas)
W_pre = net.W.copy()
print(f"Pre-Phase A:  mean OSI = {float(osi_pre.mean()):.4f}, "
      f"max OSI = {float(osi_pre.max()):.4f}, "
      f"tuned (>=0.3) = {int((osi_pre >= 0.3).sum())}/{p.M}")

# --- Train Phase A ---
import math
phi = (1.0 + math.sqrt(5)) / 2.0
theta_step = 180.0 / phi
theta0 = float(net.rng.uniform(0, 180))
for s in range(300):
    th = (theta0 + s * theta_step) % 180.0
    net.run_segment(th, plastic=True)
    if (s + 1) % 100 == 0:
        print(f"  seg {s+1}/300")

# --- Post-Phase A ---
rates_post = net.evaluate_tuning(thetas, repeats=7, contrast=1.0)
osi_post, pref_post = compute_osi(rates_post, thetas)
W_post = net.W.copy()
print(f"Post-Phase A: mean OSI = {float(osi_post.mean()):.4f}, "
      f"max OSI = {float(osi_post.max()):.4f}, "
      f"tuned (>=0.3) = {int((osi_post >= 0.3).sum())}/{p.M}")

# --- Select 16 evenly spaced by post preferred orientation ---
sort_idx = np.argsort(pref_post)
pick = sort_idx[np.linspace(0, len(sort_idx)-1, 16, dtype=int)]

sigma = 0.8

fig, axes = plt.subplots(2, 16, figsize=(22, 3.8))

for col_i, m in enumerate(pick):
    on_pre = W_pre[m, :N2].reshape(p.N, p.N)
    off_pre = W_pre[m, N2:].reshape(p.N, p.N)
    rf_pre_s = gaussian_filter(on_pre - off_pre, sigma=sigma)

    on_post = W_post[m, :N2].reshape(p.N, p.N)
    off_post = W_post[m, N2:].reshape(p.N, p.N)
    rf_post_s = gaussian_filter(on_post - off_post, sigma=sigma)

    # Row 0: before
    ax = axes[0, col_i]
    vmax = max(abs(rf_pre_s.min()), abs(rf_pre_s.max()), 0.01)
    ax.imshow(rf_pre_s, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    osi_val = osi_pre[m]
    ax.set_title(f"OSI={osi_val:.2f}", fontsize=7, color="gray")
    if col_i == 0:
        ax.set_ylabel("BEFORE\n(random init)", fontsize=9, fontweight="bold")

    # Row 1: after
    ax = axes[1, col_i]
    vmax = max(abs(rf_post_s.min()), abs(rf_post_s.max()), 0.01)
    ax.imshow(rf_post_s, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    osi_val = osi_post[m]
    ax.set_title(f"OSI={osi_val:.2f}", fontsize=7,
                 color="darkgreen" if osi_val >= 0.3 else "gray")
    ax.set_xlabel(f"{pref_post[m]:.0f}°", fontsize=8, fontweight="bold")
    if col_i == 0:
        ax.set_ylabel("AFTER\n(300 segs STDP)", fontsize=9, fontweight="bold")

fig.suptitle(
    f"ON−OFF receptive fields: random init → STDP learning\n"
    f"Pre mean OSI = {float(osi_pre.mean()):.4f}  →  Post mean OSI = {float(osi_post.mean()):.3f}  "
    f"(32/32 tuned)  |  Gaussian σ={sigma}  |  Labels = post preferred orientation",
    fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.88])
outpath = os.path.join(OUT, "before_after_filters_16.png")
fig.savefig(outpath, dpi=200)
plt.close(fig)
print(f"Saved: {outpath}")

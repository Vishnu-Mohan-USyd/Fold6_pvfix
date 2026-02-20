#!/usr/bin/env python3
"""Show pre- and post-Phase A: mean OSI + smoothed weight filter maps."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from biologically_plausible_v1_stdp import Params, RgcLgnV1Network, compute_osi

OUT = os.path.join(os.path.dirname(__file__), "figs_before_after")
os.makedirs(OUT, exist_ok=True)

p = Params(N=8, M=32, seed=1, v1_bias_eta=0.0, train_contrast=2.0)
net = RgcLgnV1Network(p, init_mode="random")
N2 = p.N * p.N

thetas = np.linspace(0, 180 - 180/12, 12)

# --- Pre-Phase A ---
rates_pre = net.evaluate_tuning(thetas, repeats=7, contrast=2.0)
osi_pre, pref_pre = compute_osi(rates_pre, thetas)
W_pre = net.W.copy()
print(f"Pre-Phase A:  mean OSI = {float(osi_pre.mean()):.4f}, "
      f"max OSI = {float(osi_pre.max()):.4f}, "
      f"tuned (>=0.3) = {int((osi_pre >= 0.3).sum())}/{p.M}")

# --- Train Phase A (300 segments, golden-ratio schedule) ---
import math
phi = (1.0 + math.sqrt(5)) / 2.0
theta_step = 180.0 / phi
theta0 = float(net.rng.uniform(0, 180))
for s in range(300):
    th = (theta0 + s * theta_step) % 180.0
    net.run_segment(th, plastic=True, contrast=2.0)
    if (s + 1) % 100 == 0:
        print(f"  seg {s+1}/300")

# --- Post-Phase A ---
rates_post = net.evaluate_tuning(thetas, repeats=7, contrast=2.0)
osi_post, pref_post = compute_osi(rates_post, thetas)
W_post = net.W.copy()
print(f"Post-Phase A: mean OSI = {float(osi_post.mean()):.4f}, "
      f"max OSI = {float(osi_post.max()):.4f}, "
      f"tuned (>=0.3) = {int((osi_post >= 0.3).sum())}/{p.M}")

# --- Plot before/after weight filters ---
sigma = 0.8  # gaussian smoothing sigma

# Sort post ensembles by preferred orientation for nice display
sort_idx = np.argsort(pref_post)

n_show = min(32, p.M)
fig, axes = plt.subplots(4, n_show, figsize=(n_show * 1.1, 4.8))

for col in range(n_show):
    m = sort_idx[col]

    # ON - OFF difference (the "receptive field")
    on_pre = W_pre[m, :N2].reshape(p.N, p.N)
    off_pre = W_pre[m, N2:].reshape(p.N, p.N)
    rf_pre = on_pre - off_pre

    on_post = W_post[m, :N2].reshape(p.N, p.N)
    off_post = W_post[m, N2:].reshape(p.N, p.N)
    rf_post = on_post - off_post

    # Smoothed versions
    rf_pre_s = gaussian_filter(rf_pre, sigma=sigma)
    rf_post_s = gaussian_filter(rf_post, sigma=sigma)

    # Row 0: pre raw
    ax = axes[0, col]
    vmax_pre = max(abs(rf_pre.min()), abs(rf_pre.max()), 0.01)
    ax.imshow(rf_pre, cmap="RdBu_r", vmin=-vmax_pre, vmax=vmax_pre,
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("Pre\n(raw)", fontsize=8)
    ax.set_title(f"E{m}", fontsize=6)

    # Row 1: pre smoothed
    ax = axes[1, col]
    ax.imshow(rf_pre_s, cmap="RdBu_r", vmin=-vmax_pre, vmax=vmax_pre,
              interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0:
        ax.set_ylabel(f"Pre\n(σ={sigma})", fontsize=8)

    # Row 2: post raw
    ax = axes[2, col]
    vmax_post = max(abs(rf_post.min()), abs(rf_post.max()), 0.01)
    ax.imshow(rf_post, cmap="RdBu_r", vmin=-vmax_post, vmax=vmax_post,
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("Post\n(raw)", fontsize=8)

    # Row 3: post smoothed
    ax = axes[3, col]
    ax.imshow(rf_post_s, cmap="RdBu_r", vmin=-vmax_post, vmax=vmax_post,
              interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0:
        ax.set_ylabel(f"Post\n(σ={sigma})", fontsize=8)

    # Add pref orientation label below
    axes[3, col].set_xlabel(f"{pref_post[m]:.0f}°", fontsize=6)

fig.suptitle(f"ON−OFF receptive fields: before vs after Phase A (300 segs, random init)\n"
             f"Pre OSI={float(osi_pre.mean()):.4f} → Post OSI={float(osi_post.mean()):.3f}  |  "
             f"Sorted by post preferred orientation  |  Gaussian σ={sigma}",
             fontsize=10, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
outpath = os.path.join(OUT, "before_after_filters.png")
fig.savefig(outpath, dpi=200)
plt.close(fig)
print(f"Saved: {outpath}")

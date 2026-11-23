"""Stationarity test on RAW (unnormalized) log-vectors.

After element-wise normalization, the mean is 0 and variance is 1 by construction,
so testing normalized data reveals nothing. The correct question is: do mu(x) and
sd(x) vary in space more than expected from the finite-sample estimation noise?

I separate estimation noise from true structure by splitting samples into two
halves A and B: noise is independent across halves, so cov(mu_A(x), mu_B(x))
across elements estimates the variance of the TRUE spatial trend (without noise).
The same applies for log sd(x).
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from src.validation import COMPS, COLORS, make_pairs, make_bins, empirical_variogram


def _split_halves(S, seed=0):
    perm = np.random.default_rng(seed).permutation(S)
    return np.sort(perm[: S // 2]), np.sort(perm[S // 2: 2 * (S // 2)])


def _signal_std(a, b):
    """sqrt(cov(a,b)) across elements; 0 if covariance is negative (just noise)."""
    c = np.mean((a - a.mean(0)) * (b - b.mean(0)), axis=0)
    return np.sqrt(np.maximum(c, 0.0)), c


def trend_report(v_train, seed=0):
    """v_train: (S,M,6) raw log-vectors (training split).

    Returns for each component:
      mean_trend_ratio : std of true mu(x) trend / pooled sd (0 = constant)
      sd_cv            : coefficient of variation of true sd(x) trend (0 = constant)
      noise_mean_ratio : reference – expected magnitude of mu(x) estimation noise = 1/sqrt(S)
    """
    S = v_train.shape[0]
    A, B = _split_halves(S, seed)
    mu_A, mu_B = v_train[A].mean(0), v_train[B].mean(0)
    lsd_A, lsd_B = np.log(v_train[A].std(0)), np.log(v_train[B].std(0))
    pooled_sd = v_train.std(0).mean(0)

    mean_sig, _ = _signal_std(mu_A, mu_B)
    sd_sig, _ = _signal_std(lsd_A, lsd_B)
    # correlation of halves: ~0 = just noise, ~1 = true physical structure
    corr = lambda a, b: np.array([np.corrcoef(a[:, c], b[:, c])[0, 1] for c in range(6)])
    return {
        "mean_trend_ratio": mean_sig / pooled_sd,
        "mean_half_corr": corr(mu_A, mu_B),
        "sd_cv": sd_sig,
        "sd_half_corr": corr(lsd_A, lsd_B),
        "noise_mean_ratio": np.full(6, 1.0 / np.sqrt(len(A))),
        "mu_A": mu_A, "mu_B": mu_B,
    }


def sample_offset_report(z_train):
    """Variance of the spatial mean of each sample (from normalized data).

    If fields lack long-range correlation, the spatial average across M elements
    is nearly zero. High variance = samples differ in "overall level".
    Roughly corresponds to 1 - variogram sill.
    """
    m = z_train.mean(axis=1)               # (S,6)
    return m.std(axis=0), m


def subregion_variograms(z_train, barycenters, axis, num_bins=20, num_pairs=300_000, seed=0):
    """Variogram computed only from pairs inside the lower/upper half of the domain along an axis."""
    i, j, d = make_pairs(barycenters, num_pairs, seed)
    bins = make_bins(d, num_bins)
    med = np.median(barycenters[:, axis])
    lo = (barycenters[i, axis] < med) & (barycenters[j, axis] < med)
    hi = (barycenters[i, axis] >= med) & (barycenters[j, axis] >= med)
    out = {}
    for name, m in (("low", lo), ("high", hi)):
        g, cnt = empirical_variogram(z_train, i[m], j[m], d[m], bins)
        g[cnt < 200] = np.nan
        out[name] = g
    return 0.5 * (bins[:-1] + bins[1:]), out


def stationarity_report(v_train, z_train, barycenters, plot_dir=None):
    tr = trend_report(v_train)
    off_std, off = sample_offset_report(z_train)
    var_diff = {}
    for ax, nm in enumerate("xyz"):
        h, g = subregion_variograms(z_train, barycenters, ax)
        var_diff[nm] = np.sqrt(np.nanmean((g["low"] - g["high"]) ** 2, axis=0))

    print("\n--- Spatial Trends & Sample Offsets ---")
    print(f"Estimation noise reference (1/sqrt(S/2)): {tr['noise_mean_ratio'][0]:.2f}")
    print(f"{'Comp':>4} | {'Mean trend/sd':>13} | {'Mean corr':>9} | {'SD CV':>7} | {'SD corr':>7} | {'Offset std':>10}")
    print("-" * 65)
    for c, n in enumerate(COMPS):
        print(
            f"{n:>4} | {tr['mean_trend_ratio'][c]:>13.3f} | {tr['mean_half_corr'][c]:>+9.2f} | "
            f"{tr['sd_cv'][c]:>7.3f} | {tr['sd_half_corr'][c]:>+7.2f} | {off_std[c]:>10.2f}"
        )

    print("\n--- Subregion Variogram RMSE (Lower vs Upper Domain Half) ---")
    header = " | ".join(f"{n:>6}" for n in COMPS)
    print(f"{'Axis':>4} | {header}")
    print("-" * 58)
    for nm, v in var_diff.items():
        vals = " | ".join(f"{x:>6.3f}" for x in v)
        print(f"{nm:>4} | {vals}")

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
        for c in (0, 3):
            axs[0].scatter(tr["mu_A"][:, c], tr["mu_B"][:, c], s=3, alpha=0.4,
                           color=COLORS[c], label=COMPS[c])
        lim = np.array(axs[0].get_xlim()); axs[0].plot(lim, lim, "k:", lw=1)
        axs[0].set_xlabel("mu(x), sample half A"); axs[0].set_ylabel("mu(x), sample half B")
        axs[0].set_title("Is the mean trend physical?\n(on diagonal = yes, cloud = just noise)"); axs[0].legend()
        axs[1].boxplot(off, tick_labels=COMPS)
        axs[1].set_title("Spatial mean of each sample\n(wide boxes = samples have varying base levels)")
        h, g = subregion_variograms(z_train, barycenters, 0)
        for c in (0, 3):
            axs[2].plot(h, g["low"][:, c], "-", color=COLORS[c], label=f"{COMPS[c]} x low")
            axs[2].plot(h, g["high"][:, c], "--", color=COLORS[c], label=f"{COMPS[c]} x high")
        axs[2].set_title("Variogram: lower vs upper half (x axis)"); axs[2].set_xlabel("h [m]")
        axs[2].legend(fontsize=7); axs[2].grid(True)
        fig.tight_layout()
        plot_path = os.path.join(plot_dir, "stationarity_raw.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"\nRaw stationarity plot saved to: {plot_path}")
    return tr, off_std, var_diff
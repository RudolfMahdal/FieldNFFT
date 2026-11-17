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

    mean_sig, mean_cov = _signal_std(mu_A, mu_B)
    sd_sig, sd_cov = _signal_std(lsd_A, lsd_B)
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

    print("\n=== RAW DATA STATIONARITY REPORT ===")
    print("1) Spatial mean trend: true trend std / pooled sd  (0 = constant)")
    print("   comp  trend/sd   halves correlation   [estimation noise = %.2f]" % tr["noise_mean_ratio"][0])
    for c, n in enumerate(COMPS):
        print(f"   {n}:   {tr['mean_trend_ratio'][c]:.3f}      {tr['mean_half_corr'][c]:+.2f}")
    print("2) Spatial variance trend: true sd(x) trend CV")
    for c, n in enumerate(COMPS):
        print(f"   {n}:   CV = {tr['sd_cv'][c]:.3f}      halves correlation {tr['sd_half_corr'][c]:+.2f}")
    print("3) Sample offset level (spatial mean std of normalized data; 0 = none)")
    print("   " + "  ".join(f"{n}={s:.2f}" for n, s in zip(COMPS, off_std)))
    print("4) Correlation structure stationarity: Variogram RMSE lower vs upper domain half")
    for nm, v in var_diff.items():
        print(f"   axis {nm}: " + "  ".join(f"{n}={x:.3f}" for n, x in zip(COMPS, v)))

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
        fig.savefig(os.path.join(plot_dir, "stationarity_raw.png"), dpi=150)
        plt.close(fig)
    return tr, off_std, var_diff
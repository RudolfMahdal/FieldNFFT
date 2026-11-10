"""Independent validation: variograms of synthetic fields vs. validation data."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from src.grf import generate_grf

COMPS = ["xx", "yy", "zz", "xy", "xz", "yz"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def make_pairs(barycenters, num_pairs=200_000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(barycenters)
    i = rng.integers(0, n, num_pairs)
    j = rng.integers(0, n, num_pairs)
    
    m = i != j
    i, j = i[m], j[m]
    return i, j, np.linalg.norm(barycenters[i] - barycenters[j], axis=1)


def make_bins(distances, num_bins=25, max_lag_fraction=0.5):
    """Linear bins up to max_lag_fraction * max distance (standard is half)."""
    return np.linspace(0.0, max_lag_fraction * distances.max(), num_bins + 1)


def empirical_variogram(z, i, j, distances, bins):
    """z: (M,6) or (S,M,6) -> (gamma (B,6), pair counts (B,)).

    For an ensemble of samples, it first averages across samples for each pair.
    """
    d2 = (z[..., i, :] - z[..., j, :]) ** 2
    if d2.ndim == 3:
        d2 = d2.mean(axis=0)
        
    B = len(bins) - 1
    b = np.digitize(distances, bins) - 1
    ok = (b >= 0) & (b < B)
    
    cnt = np.bincount(b[ok], minlength=B)
    gam = np.full((B, d2.shape[1]), np.nan)
    
    for c in range(d2.shape[1]):
        s = np.bincount(b[ok], weights=d2[ok, c], minlength=B)
        gam[cnt > 0, c] = 0.5 * s[cnt > 0] / cnt[cnt > 0]
        
    return gam, cnt


def generate_ensemble(x_shifted, specs, n_real=20, seed=1000):
    """Generates (n_real, M, 6) synthetic fields; components are independent. NO post-scaling."""
    M = len(x_shifted)
    out = np.empty((n_real, M, len(specs)))
    for r in range(n_real):
        for c, spec in enumerate(specs):
            out[r, :, c] = generate_grf(x_shifted, spec, seed=seed + len(specs) * r + c)
    return out


def variogram_report(z_train, z_val, z_synth, barycenters, plot_dir=None,
                     num_bins=25, num_pairs=200_000):
    i, j, d = make_pairs(barycenters, num_pairs)
    bins = make_bins(d, num_bins)
    centers = 0.5 * (bins[:-1] + bins[1:])

    g_tr, cnt = empirical_variogram(z_train, i, j, d, bins)
    g_va, _ = empirical_variogram(z_val, i, j, d, bins)
    g_sy = np.stack([empirical_variogram(z, i, j, d, bins)[0] for z in z_synth])
    g_sy_mean = g_sy.mean(axis=0)
    # fair band: average groups of S_val realizations (same ensemble size as validation)
    n_val = len(z_val)
    n_groups = len(g_sy) // n_val
    if n_groups >= 2:
        g_groups = g_sy[: n_groups * n_val].reshape(n_groups, n_val, *g_sy.shape[1:]).mean(axis=1)
    else:
        g_groups = g_sy
    lo, hi = np.nanpercentile(g_groups, [5, 95], axis=0)
    rmse = lambda a, b: np.sqrt(np.nanmean((a - b) ** 2, axis=0))
    metrics = {
        "rmse_synth_vs_val": rmse(g_sy_mean, g_va),
        "rmse_train_vs_val": rmse(g_tr, g_va),          # noise floor level
        "var_synth": z_synth.var(axis=1).mean(axis=0),  # mean variance per realization (should be ~1)
        "var_val": z_val.var(axis=1).mean(axis=0),
    }

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
        
        for c, ax in enumerate(axes.ravel()):
            ax.fill_between(centers, lo[:, c], hi[:, c], color=COLORS[c], alpha=0.25,
                            label="Synthetic (5–95%)")
            ax.plot(centers, g_sy_mean[:, c], "--", color=COLORS[c], label="Synthetic (Mean)")
            ax.plot(centers, g_va[:, c], "o-", color="k", ms=3, label="Validation")
            ax.plot(centers, g_tr[:, c], ":", color="gray", label="Training")
            
            ax.set_title(COMPS[c])
            ax.grid(True)
            if c == 0:
                ax.legend(fontsize=8)
                
        fig.supxlabel("Distance [m]")
        fig.supylabel(r"Semivariance $\gamma(h)$")
        fig.tight_layout()
        
        fig.savefig(os.path.join(plot_dir, "validation_variograms.png"), dpi=150)
        plt.close(fig)
        
    return centers, g_tr, g_va, g_sy, metrics
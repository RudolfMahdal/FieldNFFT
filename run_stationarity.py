import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from src.data import load_split_data, domain_setup
from src.validation import make_pairs, make_bins, empirical_variogram
from src.stationarity import stationarity_report

COMPS = ["xx", "yy", "zz", "xy", "xz", "yz"]

def exp_variogram(h: np.ndarray, sill: float, l_c: float) -> np.ndarray:
    """Theoretical exponential variogram: gamma(h) = sill * (1 - exp(-h / l_c))."""
    return sill * (1.0 - np.exp(-h / l_c))

def main():
    ZARR_URL = "s3://random-tensor-fields/homogenization_sampling_01.zarr"
    SCHEMA, GROUP = "data_schema.yaml", "tensors"
    
    # 1. Load data and split
    print("Loading data...")
    d = load_split_data(ZARR_URL, SCHEMA, GROUP, val_fraction=0.3, seed=42)
    z = d.z_train  # normalized data: (S, M, 6)
    x, L, w = domain_setup(d.barycenters)
    
    out_dir = os.path.join("plots", "stationarity")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "="*60)
    print("PART 1: STATIONARITY TEST ON RAW (UNNORMALIZED) DATA")
    print("="*60)
    
    # Reconstruct raw (unnormalized) data
    v_train_raw = d.z_train * d.sd + d.mu
    
    # Run the raw stationarity report (this will print its own results in Czech, 
    # but we will wrap it in our english pipeline)
    stationarity_report(v_train_raw, d.z_train, d.barycenters, plot_dir=out_dir)

    print("\n" + "="*60)
    print("PART 2: STATIONARITY TEST ON NORMALIZED DATA (READY FOR FINUFFT)")
    print("="*60)
    
    m = z.mean(axis=0)  # (M, 6)
    v = z.var(axis=0)   # (M, 6)
    A = np.c_[np.ones(len(x)), x / L]
    
    print("\n--- Spatial Trend of Mean and Variance ---")
    for c, name in enumerate(COMPS):
        coef = np.linalg.lstsq(A, v[:, c], rcond=None)[0]
        print(f"{name:>2}: mean(z)={m[:, c].mean():+.3f}, std(local_mean)={m[:, c].std():.3f} "
              f"(expected ~{1/np.sqrt(len(z)):.3f}), var trend (x,y,z)={coef[1:].round(3)}")

    # Auto-variograms and correlation length (l_c) extraction
    print("\n--- Fitting Correlation Lengths (l_c) ---")
    i, j, dist = make_pairs(d.barycenters, num_pairs=200_000, seed=42)
    bins = make_bins(dist, num_bins=25)
    centers = 0.5 * (bins[:-1] + bins[1:])
    
    g_auto, _ = empirical_variogram(z, i, j, dist, bins)  # (B, 6)
    
    fit_results = {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    
    for c, name in enumerate(COMPS):
        valid = ~np.isnan(g_auto[:, c])
        try:
            # Fit sill and correlation length (l_c)
            popt, _ = curve_fit(
                exp_variogram, 
                centers[valid], 
                g_auto[valid, c], 
                p0=[1.0, L / 10.0],
                bounds=([0.1, 1e-3], [2.0, L])
            )
            sill_fit, lc_fit = popt
            fit_results[name] = (sill_fit, lc_fit)
            print(f"{name:>2}: Sill = {sill_fit:.3f}, Correlation length l_c = {lc_fit:.3f} m")
        except Exception as e:
            fit_results[name] = (np.nan, np.nan)
            print(f"{name:>2}: Fit failed ({e})")

        # Plot individual components
        ax = axes.ravel()[c]
        ax.plot(centers[valid], g_auto[valid, c], 'bo-', ms=4, label='Empirical variogram')
        if not np.isnan(fit_results[name][1]):
            h_grid = np.linspace(0, centers.max(), 100)
            ax.plot(h_grid, exp_variogram(h_grid, *fit_results[name]), 'r--', 
                    label=f'Exp fit ($l_c={fit_results[name][1]:.2f}$ m)')
        ax.set_title(f'Component {name}')
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.supxlabel("Distance h [m]")
    fig.supylabel(r"Semivariance $\gamma(h)$")
    fig.tight_layout()
    
    plot_path = os.path.join(out_dir, "autovariograms_fit_normalized.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nVariogram plot saved to: {plot_path}")

    # Cross-variograms (correlations between components)
    print("\n--- Cross-Variograms (Component Correlations) ---")
    b = np.digitize(dist, bins) - 1
    ok = (b >= 0) & (b < len(bins) - 1)
    dz = z[:, i, :] - z[:, j, :]
    
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (1, 5)]
    for a, bb in pairs:
        prod = (dz[:, :, a] * dz[:, :, bb]).mean(axis=0)
        cnt = np.bincount(b[ok], minlength=len(bins) - 1)
        g_cross = 0.5 * np.bincount(b[ok], weights=prod[ok], minlength=len(bins) - 1)[cnt > 0] / cnt[cnt > 0]
        print(f"Cross {COMPS[a]}-{COMPS[bb]:<2}: gamma at bins [1, 8, 16] = {g_cross[[1, 8, 16]].round(3)}")

if __name__ == "__main__":
    main()
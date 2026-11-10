import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from src.data import load_split_data, domain_setup
from src.validation import make_pairs, make_bins, empirical_variogram

COMPS = ["xx", "yy", "zz", "xy", "xz", "yz"]

def exp_variogram(h: np.ndarray, sill: float, l_c: float) -> np.ndarray:
    """Teoretický exponenciální variogram: gamma(h) = sill * (1 - exp(-h / l_c))."""
    return sill * (1.0 - np.exp(-h / l_c))

def main():
    ZARR_URL = "s3://random-tensor-fields/homogenization_sampling_01.zarr"
    SCHEMA, GROUP = "data_schema.yaml", "tensors"
    
    # 1. Načtení dat a split
    d = load_split_data(ZARR_URL, SCHEMA, GROUP, val_fraction=0.3, seed=42)
    z = d.z_train  # normalized data: (S, M, 6)
    x, L, w = domain_setup(d.barycenters)
    
    print("=== 1. TEST PROSTOROVÉHO TRENDU (STACIONARITA VE PRŮMĚRU A ROZPTYLU) ===")
    m = z.mean(axis=0)  # (M, 6)
    v = z.var(axis=0)   # (M, 6)
    A = np.c_[np.ones(len(x)), x / L]
    
    for c, name in enumerate(COMPS):
        coef = np.linalg.lstsq(A, v[:, c], rcond=None)[0]
        print(f"{name:>2}: mean(z)={m[:, c].mean():+.3f}, std(local_mean)={m[:, c].std():.3f} "
              f"(ocakavano ~{1/np.sqrt(len(z)):.3f}), var trend (x,y,z)={coef[1:].round(3)}")

    # 2. AUTO-VARIOGRAMY A URČENÍ KORELAČNÍCH DÉLEK (l_c)
    print("\n=== 2. FITOVÁNÍ KORELAČNÍCH DÉLEK (l_c) ===")
    i, j, dist = make_pairs(d.barycenters, num_pairs=200_000, seed=42)
    bins = make_bins(dist, num_bins=25)
    centers = 0.5 * (bins[:-1] + bins[1:])
    
    g_auto, _ = empirical_variogram(z, i, j, dist, bins)  # (B, 6)
    
    fit_results = {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    
    for c, name in enumerate(COMPS):
        valid = ~np.isnan(g_auto[:, c])
        try:
            # Fitování prahu (sill) a korelační délky (l_c)
            popt, _ = curve_fit(
                exp_variogram, 
                centers[valid], 
                g_auto[valid, c], 
                p0=[1.0, L / 10.0],
                bounds=([0.1, 1e-3], [2.0, L])
            )
            sill_fit, lc_fit = popt
            fit_results[name] = (sill_fit, lc_fit)
            print(f"{name:>2}: Sill = {sill_fit:.3f}, Korelační délka l_c = {lc_fit:.3f} m")
        except Exception as e:
            fit_results[name] = (np.nan, np.nan)
            print(f"{name:>2}: Fitování selhalo ({e})")

        # Plot jednotlivých komponent
        ax = axes.ravel()[c]
        ax.plot(centers[valid], g_auto[valid, c], 'bo-', ms=4, label='Empirický variogram')
        if not np.isnan(fit_results[name][1]):
            h_grid = np.linspace(0, centers.max(), 100)
            ax.plot(h_grid, exp_variogram(h_grid, *fit_results[name]), 'r--', 
                    label=f'Exp fit ($l_c={fit_results[name][1]:.2f}$ m)')
        ax.set_title(f'Komponenta {name}')
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.supxlabel("Vzdálenost h [m]")
    fig.supylabel(r"Semivariance $\gamma(h)$")
    fig.tight_layout()
    
    out_dir = os.path.join("plots", "stationarity")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "autovariograms_fit.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nGraf variogramů uložen do: {plot_path}")

    # 3. KŘÍŽOVÉ VARIOGRAMY (CROSS-VARIOGRAMS)
    print("\n=== 3. KŘÍŽOVÉ VARIOGRAMY (KORELACE MEZI KOMPONENTAMI) ===")
    b = np.digitize(dist, bins) - 1
    ok = (b >= 0) & (b < len(bins) - 1)
    dz = z[:, i, :] - z[:, j, :]
    
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (1, 5)]
    for a, bb in pairs:
        prod = (dz[:, :, a] * dz[:, :, bb]).mean(axis=0)
        cnt = np.bincount(b[ok], minlength=len(bins) - 1)
        g_cross = 0.5 * np.bincount(b[ok], weights=prod[ok], minlength=len(bins) - 1)[cnt > 0] / cnt[cnt > 0]
        print(f"Cross {COMPS[a]}-{COMPS[bb]:<2}: gamma u binů [1, 8, 16] = {g_cross[[1, 8, 16]].round(3)}")

if __name__ == "__main__":
    main()
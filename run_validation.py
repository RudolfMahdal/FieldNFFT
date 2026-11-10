import os

from src.data import load_split_data, domain_setup
from src.spectral import fit_empirical_spectra
import numpy as np
from src.validation import (generate_ensemble, variogram_report,
                            make_pairs, make_bins, empirical_variogram)
def main():
    ZARR_URL = "s3://random-tensor-fields/homogenization_sampling_01.zarr"
    SCHEMA, GROUP = "data_schema.yaml", "tensors"
    N_FREQ, N_REAL, VAL_FRACTION, SEED = 64, 40, 0.3, 7

    plot_dir = os.path.join("plots", ZARR_URL.split("/")[-1], "validation")

    print(f"Loading data from {ZARR_URL}...")
    d = load_split_data(ZARR_URL, SCHEMA, GROUP, VAL_FRACTION, SEED)
    print(f"Split: train={len(d.train_idx)}, validation={len(d.val_idx)}")

    x, L, w = domain_setup(d.barycenters)
    print(f"Physical domain size L = {L:.3f}")

    print("Estimating spectrum from TRAINING data only (non-parametric)...")
    specs = fit_empirical_spectra(x, d.z_train, w, L, N_FREQ, target_variance=1.0)

    print(f"Generating synthetic fields ({N_REAL} realizations, FINUFFT type 2)...")
    z_synth = generate_ensemble(x, specs, n_real=N_REAL)

        # nugget calibration from TRAINING data only
    i, j, dist = make_pairs(d.barycenters)       # same pairs/bins as variogram_report
    bins = make_bins(dist)
    g_tr, cnt = empirical_variogram(d.z_train, i, j, dist, bins)
    g_sy = np.mean([empirical_variogram(z, i, j, dist, bins)[0] for z in z_synth], axis=0)
    k0 = int(np.flatnonzero(cnt > 0)[0])         # first non-empty bin
    tau = np.clip(g_tr[k0] - g_sy[k0], 0.0, None)  # (6,) nugget variance per component
    print(f"  nugget tau per component (bin {k0}):", np.round(tau, 3))
    rng = np.random.default_rng(123)
    z_synth = z_synth + rng.standard_normal(z_synth.shape) * np.sqrt(tau)
    
    print("Computing variograms and comparing with validation data...")
    _, _, _, _, m = variogram_report(d.z_train, d.z_val, z_synth, d.barycenters, plot_dir)

    comps = ["xx", "yy", "zz", "xy", "xz", "yz"]
    print("\nVariogram RMSE (lower = better):")
    for c, name in enumerate(comps):
        print(f"  {name}: synth vs val = {m['rmse_synth_vs_val'][c]:.4f} | "
              f"train vs val (noise floor) = {m['rmse_train_vs_val'][c]:.4f} | "
              f"var synth = {m['var_synth'][c]:.3f}, var val = {m['var_val'][c]:.3f}")

    print(f"\nFull validation plot saved to: {plot_dir}/validation_variograms.png")


if __name__ == "__main__":
    main()
import os
import numpy as np
import matplotlib.pyplot as plt
from src.data import load_split_data, domain_setup
from src.spectral import fit_empirical_spectra
from src.validation import generate_ensemble

def main():
    ZARR_URL = "s3://random-tensor-fields/homogenization_sampling_01.zarr"
    SCHEMA, GROUP = "data_schema.yaml", "tensors"
    
    print("Loading data...")
    d = load_split_data(ZARR_URL, SCHEMA, GROUP, val_fraction=0.3, seed=7)
    x, L, w = domain_setup(d.barycenters)
    
    print("Calculating spectra and cross-correlations (Cholesky)...")
    specs = fit_empirical_spectra(x, d.z_train, w, L, N_freq=64, target_variance=1.0)
    
    # Cholesky matrix from training data (binds tensor components for physical consistency)
    z_flat = d.z_train.reshape(-1, 6)
    L_matrix = np.linalg.cholesky(np.corrcoef(z_flat, rowvar=False))
    
    print("Generating 1 synthetic sample...")
    z_synth = generate_ensemble(x, specs, n_real=1, seed=42, L_matrix=L_matrix)
    
    # Denormalization - returning to raw physical values
    print("Denormalizing data (returning to physical values)...")
    v_synth_raw = z_synth[0] * d.sd + d.mu
    v_real_raw = d.z_val[0] * d.sd + d.mu  # First sample from the validation set
    
    # Plotting
    print("Plotting 3D comparison...")
    out_dir = os.path.join("plots", "visualization")
    os.makedirs(out_dir, exist_ok=True)
    
    # Plotting the tensile component xx and the shear component xy
    fig = plt.figure(figsize=(16, 10))
    px, py, pz = x[:, 0], x[:, 1], x[:, 2]
    
    def plot_scatter(ax_idx, data, title, cmap='viridis'):
        ax = fig.add_subplot(2, 2, ax_idx, projection='3d')
        scatter = ax.scatter(px, py, pz, c=data, cmap=cmap, s=5, alpha=0.8)
        fig.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
        ax.set_title(title)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')

    # Real vs Synth for 'xx'
    plot_scatter(1, v_real_raw[:, 0], "REALITY (Validation) - Component XX", 'viridis')
    plot_scatter(2, v_synth_raw[:, 0], "YOUR GENERATOR (Synthetic) - Component XX", 'viridis')
    
    # Real vs Synth for 'xy'
    plot_scatter(3, v_real_raw[:, 3], "REALITY (Validation) - Component XY (Shear)", 'plasma')
    plot_scatter(4, v_synth_raw[:, 3], "YOUR GENERATOR (Synthetic) - Component XY (Shear)", 'plasma')

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "real_vs_synth_3d.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Done! Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
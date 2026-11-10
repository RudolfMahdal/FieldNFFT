import os
import numpy as np
import matplotlib.pyplot as plt
from src.tensor_funcs import comps_to_matrix, tensor_to_vector, fit_normalization, normalize
from src.zarrfuse_load import load_fuse

def load_and_preprocess_data(zarr_url, schema, group):
    """Loads Zarr data, filters valid samples, and normalizes tensors."""
    print(f"Loading data from: {zarr_url} ...")
    ds = load_fuse(schema, group)
    ds_valid = ds.where((ds.return_code == 0).compute(), drop=True)
    
    n_samples = ds_valid.sizes['sample_id']
    print(f"Valid samples count: {n_samples}")
    print("Available data variables:", list(ds_valid.data_vars))
    
    cond_comps = ds_valid.conductivity.transpose("sample_id", "el_id", "tensor_comp").values
    barycenters = ds.el_barycenter.transpose("el_id", "xyz").values 
    
    T_matrices = comps_to_matrix(cond_comps)
    cond_vectors = tensor_to_vector(T_matrices)
    
    mu, sd = fit_normalization(cond_vectors, len(cond_vectors))
    z_normalized = normalize(cond_vectors, mu, sd)
    
    return barycenters, cond_comps, cond_vectors, z_normalized, n_samples


def plot_3d_components(barycenters, cond_comps, sample_idx, save_dir):
    """Plots the 3D spatial distribution of all 6 tensor components for a specific sample."""
    if sample_idx == "all":
        print("Skipping 3D plot for 'all' samples (must specify a single sample).")
        return

    x, y, z = barycenters[:, 0], barycenters[:, 1], barycenters[:, 2]
    comp_names = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
    current_comps = cond_comps[sample_idx]
    
    fig = plt.figure(figsize=(18, 10))
    for i in range(6):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        scatter = ax.scatter(x, y, z, c=current_comps[:, i], cmap='viridis', s=5, alpha=0.8)
        fig.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
        ax.set_title(f'Component {comp_names[i]}')
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')

    plt.suptitle(f'3D Spatial Distribution of Tensor Components (Sample {sample_idx})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3d_components_distribution.png"), bbox_inches='tight', dpi=150)
    plt.close(fig)




"""Data loading, train/validation split, and normalization.

Normalization (mean and standard deviation per element and component) is estimated 
ONLY from the training samples and then applied to both the training and validation sets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from src.tensor_funcs import comps_to_matrix, tensor_to_vector, normalize


@dataclass
class SplitData:
    barycenters: np.ndarray      # (M, 3) original coordinates
    train_idx: np.ndarray        # indices of training samples
    val_idx: np.ndarray          # indices of validation samples
    mu: np.ndarray               # (M, 6) estimated from the training set
    sd: np.ndarray               # (M, 6) estimated from the training set
    z_train: np.ndarray          # (S_train, M, 6) normalized training set
    z_val: np.ndarray            # (S_val, M, 6) normalized validation set


def split_indices(n_samples: int, val_fraction: float = 0.3, seed: int = 0):
    """Random (reproducible) splitting of sample indices."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    n_val = max(1, int(round(n_samples * val_fraction)))
    if n_val >= n_samples:
        raise ValueError("Too few samples to perform a split.")
    perm = np.random.default_rng(seed).permutation(n_samples)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def build_split_data(barycenters, cond_comps, val_fraction=0.3, seed=0) -> SplitData:
    """Pure function (no I/O): tensors -> vectors -> split -> normalize from train."""
    cond_vectors = tensor_to_vector(comps_to_matrix(cond_comps))
    train_idx, val_idx = split_indices(len(cond_vectors), val_fraction, seed)
    
    train = cond_vectors[train_idx]
    mu, sd = train.mean(axis=0), train.std(axis=0)
    
    if np.any(sd <= 0):
        raise ValueError("Zero standard deviation found in some element/component.")
        
    return SplitData(
        barycenters=barycenters,
        train_idx=train_idx,
        val_idx=val_idx,
        mu=mu,
        sd=sd,
        z_train=normalize(train, mu, sd),
        z_val=normalize(cond_vectors[val_idx], mu, sd),
    )

def load_split_data(zarr_url, schema, group, val_fraction=0.3, seed=0) -> SplitData:
    """Loads a Zarr dataset and splits it into training and validation sets."""
    from src.zarrfuse_load import load_fuse, setup_environment

    setup_environment(zarr_url)
    ds = load_fuse(schema, group)
    ds_valid = ds.where((ds.return_code == 0).compute(), drop=True)
    
    cond_comps = ds_valid.conductivity.transpose("sample_id", "el_id", "tensor_comp").values
    barycenters = ds.el_barycenter.transpose("el_id", "xyz").values
    print(f"Valid samples found: {len(cond_comps)}")
    
    return build_split_data(barycenters, cond_comps, val_fraction, seed)


def domain_setup(barycenters):
    """Shifts coordinates to origin, calculates domain size L and integration weights (sum = L^3).

    Weights: d_nn^3 (rough estimate of element volume). If the dataset contains 
    exact element volumes, it would be better to use those instead.
    """
    shifted = barycenters - barycenters.min(axis=0)
    L = float(shifted.max())
    
    d, _ = cKDTree(barycenters).query(barycenters, k=2)
    w = d[:, 1] ** 3
    
    # Scale weights so they perfectly equal the domain volume L^3 to prevent overflow
    w *= (L ** 3) / w.sum()
    
    return shifted, L, w
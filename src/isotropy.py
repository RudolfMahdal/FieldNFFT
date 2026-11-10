"""src/isotropy.py – Funkce pro test izotropie a normality Fourierových koeficientů."""
from __future__ import annotations

import finufft
import numpy as np
from scipy.stats import shapiro


def compute_directional_spectra(x_shifted, z_train, w, L, N_freq=64):
    """Krok 4: Počítá 1D spektrální hustoty podél os X, Y, Z pro ověření izotropie."""
    x_nu = [(x_shifted[:, i] / L) * 2.0 * np.pi - np.pi for i in range(3)]
    
    acc_spec = np.zeros((N_freq, N_freq, N_freq))
    n_samples, n_elems, n_comps = z_train.shape
    
    for s in range(n_samples):
        for c in range(n_comps):
            z = z_train[s, :, c]
            f_hat = finufft.nufft3d1(*x_nu, (z * w).astype(np.complex128), (N_freq, N_freq, N_freq))
            P = np.abs(f_hat) ** 2 / L ** 3
            P -= np.sum((w * z) ** 2) / L ** 3  # Odečtení shot-noise
            acc_spec += P

    P_mean = acc_spec / (n_samples * n_comps)
    
    spec_x = P_mean.mean(axis=(1, 2))
    spec_y = P_mean.mean(axis=(0, 2))
    spec_z = P_mean.mean(axis=(0, 1))
    k_axis = np.arange(-N_freq // 2, N_freq // 2) * (2.0 * np.pi / L)
    
    return k_axis, spec_x, spec_y, spec_z


def test_fourier_normality(x_shifted, z_train, w, L, N_freq=32, max_samples=5000):
    """Krok 5: Seskupí Fourierovy koeficienty do radiálních slupek a otestuje Gaussovo rozdělení."""
    x_nu = [(x_shifted[:, i] / L) * 2.0 * np.pi - np.pi for i in range(3)]
    k = np.arange(-N_freq // 2, N_freq // 2)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    kmag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2).ravel()
    
    n_samples = z_train.shape[0]
    coeffs_list = []
    for s in range(n_samples):
        z = z_train[s, :, 0]  # složka xx
        f_hat = finufft.nufft3d1(*x_nu, (z * w).astype(np.complex128), (N_freq, N_freq, N_freq))
        coeffs_list.append(f_hat.ravel())
        
    coeffs_matrix = np.array(coeffs_list)
    shells = np.rint(kmag).astype(int)
    unique_shells = np.unique(shells)
    unique_shells = unique_shells[(unique_shells > 0) & (unique_shells < N_freq // 2)]
    
    p_real, p_imag, radial_k = [], [], []
    rng = np.random.default_rng(42)
    
    for s_idx in unique_shells:
        idx = np.where(shells == s_idx)[0]
        coeffs = coeffs_matrix[:, idx].ravel()
        if len(coeffs) > max_samples:
            coeffs = rng.choice(coeffs, size=max_samples, replace=False)
            
        _, p_r = shapiro(coeffs.real)
        _, p_i = shapiro(coeffs.imag)
        
        radial_k.append(s_idx * (2.0 * np.pi / L))
        p_real.append(p_r)
        p_imag.append(p_i)
        
    return np.array(radial_k), np.array(p_real), np.array(p_imag)
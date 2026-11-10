"""Non-parametric spectrum estimation from data (replacement for Gaussian/Exponential models).

Process:
  1. For each training sample and component: NUFFT Type 1 -> periodogram.
  2. Average across samples and components (reduces periodogram noise).
  3. Radial average in shells of |k| = 1 (units of 2π/L) -> profile S(|ω|).
  4. EmpiricalSpectrum shares the same interface as GaussianCorrelation
     (L, N_freq, dim, f_points, spectral_density), making it fully compatible 
     with grf.generate_grf.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import finufft
import numpy as np


def _k_grid(N_freq: int):
    k = np.arange(-N_freq // 2, N_freq // 2)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    return np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)


def periodogram_3d(x_shifted, z, w, L, N_freq, subtract_diagonal=True):
    """Periodogram |Σ w_j z_j e^{-iωx_j}|² / L³ for z of shape (M,) -> (N,N,N).

    The squared sum contains a diagonal term Σ w_j² z_j² (point sampling on an
    irregular grid). This term is constant across all frequencies ("shot noise") and 
    with N_freq³ modes it would overpower the actual variance. Therefore, it is 
    standard practice to subtract it (exactly for the given realization); leaving 
    only the cross terms. After subtraction, the periodogram can be negative in 
    individual modes.
    """
    x_nu = [(x_shifted[:, i] / L) * 2.0 * np.pi - np.pi for i in range(3)]
    f_hat = finufft.nufft3d1(*x_nu, (z * w).astype(np.complex128),
                             (N_freq, N_freq, N_freq))
    P = np.abs(f_hat) ** 2 / L ** 3
    
    if subtract_diagonal:
        P = P - np.sum((w * z) ** 2) / L ** 3
        
    return P


def mean_periodogram(x_shifted, z_samples, w, L, N_freq=64, subtract_diagonal=True):
    """Averages periodograms across all samples (S,M,6) and components."""
    acc = np.zeros((N_freq,) * 3)
    n = 0
    for s in range(z_samples.shape[0]):
        for c in range(z_samples.shape[2]):
            acc += periodogram_3d(x_shifted, z_samples[s, :, c], w, L, N_freq,
                                  subtract_diagonal)
            n += 1
    return acc / n


def radial_profile(P, N_freq):
    """Radial average of the 3D spectrum into shells of width 1 (in units of 2π/L).

    Returns (k_centers, S_mean, counts); where k_centers is the average |k| inside the shell.
    """
    kmag = _k_grid(N_freq)
    shell = np.rint(kmag).astype(int).ravel()
    p = P.ravel()
    
    cnt = np.bincount(shell)
    keep = cnt > 0
    
    s_mean = np.bincount(shell, weights=p)[keep] / cnt[keep]
    k_mean = np.bincount(shell, weights=kmag.ravel())[keep] / cnt[keep]
    
    return k_mean, s_mean, cnt[keep]


@dataclass
class EmpiricalSpectrum:
    """Radial spectral density defined by a lookup table (interpolates in log S)."""

    L: float
    N_freq: int
    k_nodes: np.ndarray            # |k| in units of 2π/L
    S_nodes: np.ndarray            # S at nodes (>0)
    dim: int = 3
    f_points: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        k = np.arange(-self.N_freq // 2, self.N_freq // 2)
        self.f_points = k * (2.0 * np.pi / self.L)
        self.S_nodes = np.maximum(np.asarray(self.S_nodes, float), 1e-300)

    def spectral_density(self, omega_mag):
        k = np.asarray(omega_mag) * self.L / (2.0 * np.pi)
        return np.exp(np.interp(k, self.k_nodes, np.log(self.S_nodes)))

    def implied_variance(self):
        """σ² implied by the spectrum: (1/L³) Σ_k S(ω_k)."""
        kmag = _k_grid(self.N_freq)
        S = self.spectral_density(kmag * 2.0 * np.pi / self.L)
        return S.sum() / self.L ** 3


def fit_empirical_spectrum(x_shifted, z_train, w, L, N_freq=64,
                           target_variance=1.0):
    """Estimates EmpiricalSpectrum from training data.

    target_variance: the spectrum is rescaled so that (1/L³)ΣS = target_variance.
    The data is unit-normalized, so it should be 1.0. The scaling corrects for the 
    fact that the mesh might not perfectly fill the entire [0,L]³ cube (which would 
    otherwise underestimate the spectrum). Passing None skips scaling.
    """
    P = mean_periodogram(x_shifted, z_train, w, L, N_freq)
    k, S, _ = radial_profile(P, N_freq)

    # noise floor from the outermost shells (pure noise there): mean = bias of the
    # shot-noise subtraction, std = scatter
    outer = k > 0.6 * k[-1]
    base, sig = S[outer].mean(), S[outer].std()
    S = S - base

    # cut at the first shell that is not clearly above the noise
    cut = int(np.flatnonzero(S < 3.0 * sig)[0])

    # exponential tail fitted on the last shells before the cut
    lo = max(cut - 5, 1)
    slope = np.polyfit(k[lo:cut], np.log(S[lo:cut]), 1)[0]
    slope = min(slope, -0.05)                      # must decay
    k_tail = k[cut - 1] + np.arange(1.0, 60.0)
    S_tail = S[cut - 1] * np.exp(slope * (k_tail - k[cut - 1]))

    print(f"  noise base={base:.3g}, sigma={sig:.3g}, cut at shell {cut}, "
          f"tail slope={slope:.3f}/shell")

    k = np.concatenate([k[:cut], k_tail])
    S = np.concatenate([S[:cut], S_tail])
    spec = EmpiricalSpectrum(L=L, N_freq=N_freq, k_nodes=k, S_nodes=S)
    
    if target_variance is not None:
        spec.S_nodes = spec.S_nodes * (target_variance / spec.implied_variance())
        
    return spec

def fit_empirical_spectra(x_shifted, z_train, w, L, N_freq=64, target_variance=1.0):
    """One EmpiricalSpectrum per tensor component (list of 6)."""
    return [
        fit_empirical_spectrum(x_shifted, z_train[:, :, c:c + 1], w, L, N_freq,
                               target_variance)
        for c in range(z_train.shape[2])
    ]
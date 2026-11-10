"""Functions for working with tensors (symmetric positive-definite 3x3 matrices) and their 6-vectors.

Full transformation path:
    tensor (3x3) --log--> symmetric matrix --Mandel--> vector (6) --normalization--> z
    z --back--> vector --back--> matrix --exp--> tensor

Component order: xx, yy, zz, xy, xz, yz  
All functions support tensor arrays of shape (..., 3, 3) or (..., 6).
"""
import numpy as np

IDX = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]       # Matrix indices for components
W = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])   # Mandel weights (√2 for off-diagonals)


# --- 6 components <-> 3x3 matrix ---------------------------------------------------
def comps_to_matrix(c):
    """Converts (..., 6) array to (..., 3, 3) symmetric matrices."""
    c = np.asarray(c, dtype=float)
    M = np.zeros(c.shape[:-1] + (3, 3))
    for k, (i, j) in enumerate(IDX):
        M[..., i, j] = c[..., k]
        M[..., j, i] = c[..., k]
    return M


def matrix_to_comps(M):
    """Extracts components from (..., 3, 3) matrices to (..., 6) array."""
    return np.stack([M[..., i, j] for i, j in IDX], axis=-1)


# --- Matrix logarithm and exponential ----------------------------------------------
def matrix_log(T):
    """Matrix logarithm for a positive-definite matrix (via eigendecomposition)."""
    lam, Q = np.linalg.eigh(T)
    return (Q * np.log(lam)[..., None, :]) @ np.swapaxes(Q, -1, -2)


def matrix_exp(L):
    """Matrix exponential for a symmetric matrix; output is always positive-definite."""
    lam, Q = np.linalg.eigh(L)
    return (Q * np.exp(lam)[..., None, :]) @ np.swapaxes(Q, -1, -2)


# --- Tensor <-> Vector (log + Mandel) ----------------------------------------------
def tensor_to_vector(T):
    """(..., 3, 3) tensor -> (..., 6) vector: applies logarithm, then Mandel weights (√2 for off-diagonals)."""
    return matrix_to_comps(matrix_log(T)) * W


def vector_to_tensor(v):
    """(..., 6) vector -> (..., 3, 3) tensor; inverse of tensor_to_vector."""
    return matrix_exp(comps_to_matrix(np.asarray(v) / W))


# --- Normalization ------------------------------------------------------------------
def fit_normalization(x, n_train):
    """Calculates mean and std from the first n_train samples.
    x has shape (samples, elements, 6); output is (elements, 6).
    """
    return x[:n_train].mean(axis=0), x[:n_train].std(axis=0)


def normalize(x, mu, sd):
    """Normalizes the input array using provided mean and std."""
    return (x - mu) / sd


def denormalize(z, mu, sd):
    """Reverts normalization on array z using provided mean and std."""
    return z * sd + mu
"""Funkce pro práci s tenzory (symetrické kladně definitní matice 3x3) a jejich 6-vektory.

Celá cesta tam a zpět:
    tenzor (3x3) --log--> symetrická matice --Mandel--> vektor (6) --normalizace--> z
    z --zpět--> vektor --zpět--> matice --exp--> tenzor

Pořadí složek : xx, yy, zz, xy, xz, yz  
Všechny funkce fungují i na polích tenzorů, tvar (..., 3, 3) nebo (..., 6).
"""
import numpy as np

IDX = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]       # které prvky matice jsou složky
W = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])   # Mandelova váha (√2 u mimodiagonál)


# --- 6 složek <-> matice 3x3 ---------------------------------------------------
def comps_to_matrix(c):
    """(..., 6) -> (..., 3, 3) symetrická matice."""
    c = np.asarray(c, dtype=float)
    M = np.zeros(c.shape[:-1] + (3, 3))
    for k, (i, j) in enumerate(IDX):
        M[..., i, j] = c[..., k]
        M[..., j, i] = c[..., k]
    return M


def matrix_to_comps(M):
    """(..., 3, 3) -> (..., 6)."""
    return np.stack([M[..., i, j] for i, j in IDX], axis=-1)


# --- logaritmus a exponenciála ----------------------------------------------------
def matrix_log(T):
    """Maticový logaritmus kladně definitní matice (přes vlastní rozklad)."""
    lam, Q = np.linalg.eigh(T)
    return (Q * np.log(lam)[..., None, :]) @ np.swapaxes(Q, -1, -2)


def matrix_exp(L):
    """Maticová exponenciála symetrické matice; výsledek je vždy kladně definitní."""
    lam, Q = np.linalg.eigh(L)
    return (Q * np.exp(lam)[..., None, :]) @ np.swapaxes(Q, -1, -2)


# --- tenzor <-> vektor (log + Mandel) -----------------------------------------------
def tensor_to_vector(T):
    """(..., 3, 3) tenzor -> (..., 6) vektor: logaritmus, pak složky s √2 u mimodiagonál."""
    return matrix_to_comps(matrix_log(T)) * W


def vector_to_tensor(v):
    """(..., 6) vektor -> (..., 3, 3) tenzor; inverzní k tensor_to_vector."""
    return matrix_exp(comps_to_matrix(np.asarray(v) / W))


# --- normalizace ------------------------------------------------------------------
def fit_normalization(x, n_train):
    """Průměr a sd z prvních n_train vzorků. x má tvar (vzorky, elementy, 6); výstup (elementy, 6)."""
    return x[:n_train].mean(axis=0), x[:n_train].std(axis=0)


def normalize(x, mu, sd):
    return (x - mu) / sd


def denormalize(z, mu, sd):
    return z * sd + mu
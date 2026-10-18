"""Zkouška tenzorových funkcí na malých vymyšlených datech (bez načítání z úložiště)."""
import numpy as np

from tensor_funcs import (comps_to_matrix, matrix_to_comps, tensor_to_vector, vector_to_tensor,
                          fit_normalization, normalize, denormalize)

# --- 1) jeden tenzor ------------------------------------------------------------------
# 6 složek v pořadí xx, yy, zz, xy, xz, yz
comps = np.array([4.0, 2.0, 1.0, 0.5, 0.0, 0.3])
T = comps_to_matrix(comps)
print("Matice 3x3:\n", T)
print("Vlastní čísla:", np.linalg.eigvalsh(T).round(3))

v = tensor_to_vector(T)
print("\nVektor (log + Mandel):", v.round(3))

T_back = vector_to_tensor(v)
print("Zpět na tenzor, max chyba:", np.abs(T_back - T).max())

# --- 2) pole tenzorů: (vzorky, elementy, 3, 3) -------------------------------------------
rng = np.random.default_rng(0)
A = rng.normal(size=(10, 4, 3, 3))
tensors = A @ np.swapaxes(A, -1, -2) + 0.5 * np.eye(3)       # 10 vzorků, 4 elementy, kladně definitní
print("\nTvar pole tenzorů:", tensors.shape)

x = tensor_to_vector(tensors)
print("Tvar vektorů:", x.shape)                               # (10, 4, 6)

# --- 3) normalizace podle trénovacích vzorků (prvních 8) -------------------------------
n_train = 8
mu, sd = fit_normalization(x, n_train)
z = normalize(x, mu, sd)
print("\nTrénovací z: průměr =", z[:n_train].mean().round(6), ", sd =", z[:n_train].std().round(3))

# --- 4) celá cesta zpět ------------------------------------------------------------------
tensors_back = vector_to_tensor(denormalize(z, mu, sd))
print("Celá cesta tam a zpět, max chyba:", np.abs(tensors_back - tensors).max())

# S pravdivými daty by se místo vymyšlených tenzorů použilo například:
#   from zarrfuse_load import load_fuse
#   ds = load_fuse("tensor_schema.yaml")
#   tensors = comps_to_matrix(ds["value"].values)             # (s_id, el_id, 3, 3)
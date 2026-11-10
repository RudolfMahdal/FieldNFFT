"""run_isotropy_fourier.py – Executable skript pro spuštění bodů 4 a 5."""
import os
import matplotlib.pyplot as plt
import numpy as np
from src.data import load_split_data, domain_setup
from src.isotropy import compute_directional_spectra, test_fourier_normality

ZARR_URL = "s3://random-tensor-fields/homogenization_sampling_01.zarr"
SCHEMA, GROUP = "data_schema.yaml", "tensors"

def main():
    print("Načítání dat...")
    d = load_split_data(ZARR_URL, SCHEMA, GROUP, val_fraction=0.3, seed=42)
    x, L, w = domain_setup(d.barycenters)
    
    out_dir = os.path.join("plots", "isotropy_fourier")
    os.makedirs(out_dir, exist_ok=True)
    
    # --- KROK 4: Test Izotropie ---
    print("\n=== KROK 4: SPANÍ SMĚROVÝCH SPEKTR (IZOTROPIE) ===")
    k_axis, spec_x, spec_y, spec_z = compute_directional_spectra(x, d.z_train, w, L, N_freq=64)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    half = len(k_axis) // 2
    k_pos = k_axis[half:]
    
    # Oříznutí záporných hodnot z odečtení šumu na floor 1e-3
    ax.semilogy(k_pos, np.maximum(spec_x[half:], 1e-3), 'r-', label='Spektrum X ($k_x$)')
    ax.semilogy(k_pos, np.maximum(spec_y[half:], 1e-3), 'g--', label='Spektrum Y ($k_y$)')
    ax.semilogy(k_pos, np.maximum(spec_z[half:], 1e-3), 'b:', label='Spektrum Z ($k_z$)')
    
    ax.set_xlim(0, 0.5)  # Detail na oblast, kde je fyzikální signál
    ax.set_ylim(1e-3, 100)
    ax.set_xlabel('Úhlová frekvence $k$ [rad/m]')
    ax.set_ylabel('Spektrální hustota $S(k)$')
    ax.set_title('Test izotropie: 1D spektrální řezy v osách X, Y, Z')
    ax.grid(True, which="both", ls="--")
    ax.legend()

    fig.savefig(os.path.join(out_dir, "directional_isotropy.png"), dpi=150)
    plt.close(fig)
    print(f"Graf izotropie uložen do: {out_dir}/directional_isotropy.png")

    # --- KROK 5: Test Normality ---
    print("\n=== KROK 5: TEST NORMALITY FOURIEROVÝCH KOEFICIENTŮ ===")
    rad_k, p_real, p_imag = test_fourier_normality(x, d.z_train, w, L, N_freq=32)
    
    print(f"Průměrná p-hodnota (Re): {p_real.mean():.4f}")
    print(f"Průměrná p-hodnota (Im): {p_imag.mean():.4f}")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rad_k, p_real, 'bo-', label='Re(f_hat) p-val')
    ax.plot(rad_k, p_imag, 'rs--', label='Im(f_hat) p-val')
    ax.axhline(0.05, color='r', linestyle=':', label='Hladina významnosti alpha = 0.05')
    ax.set_xlabel('Radiální frekvence $|k|$ [rad/m]')
    ax.set_ylabel('Shapiro-Wilk p-value')
    ax.set_title('Test normality Fourierových koeficientů podle radiálních slupek')
    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(out_dir, "fourier_normality.png"), dpi=150)
    plt.close(fig)
    print(f"Graf normality uložen do: {out_dir}/fourier_normality.png")

if __name__ == "__main__":
    main()
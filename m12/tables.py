import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants
c = 3e8  # Speed of light in m/s
k_B = 1.38e-23  # Boltzmann constant in J/K
T = 290  # Ambient temperature in Kelvin

# Get user input for resonant frequency
freq_ghz = float(input("Enter the resonant frequency of the sensor (in GHz): "))
freq_hz = freq_ghz * 1e9
wavelength = c / freq_hz  # Wavelength in meters

print(f"Calculated wavelength: {wavelength:.3e} m")

# Fixed parameters
Gs_dBi = 5.5  # UWB antenna gain in dBi
S11_dB = -10  # Reflection coefficient (dB)
Ld_dB = -0.5  # Delay line loss (dB)
B = 1e9  # Bandwidth in Hz
SNR_dB = 20  # Required SNR in dB


# Conversion functions
def db_to_linear(db):
    return 10 ** (db / 10)


# Convert fixed values to linear
Gs = db_to_linear(Gs_dBi)
S11 = db_to_linear(S11_dB)
Ld = db_to_linear(Ld_dB)
SNR = db_to_linear(SNR_dB)
E = k_B * T

# Decision table settings
Pt_dBm_values = [-10, -5, 0, 5, 10]
GtGr_dBi_values = [0, 5, 10, 15, 20]
F_dB_values = [0, 1, 3, 6, 10]  # Noise figure in dB

for F_dB in F_dB_values:
    F = db_to_linear(F_dB)
    P_min = E * B * F * SNR  # In Watts
    table = []

    for Pt_dBm in Pt_dBm_values:
        row = []
        Pt = db_to_linear(Pt_dBm - 30)  # dBm to Watts
        for GtGr_dBi in GtGr_dBi_values:
            Gt = db_to_linear(GtGr_dBi)
            Gr = db_to_linear(GtGr_dBi)
            const_term = Pt * Gt * Gr * Gs**2 * S11 * Ld
            d_max = wavelength / (4 * np.pi) * (const_term / P_min) ** 0.25
            row.append(d_max)
        table.append(row)

    df = pd.DataFrame(
        table,
        columns=[f"{g} dBi" for g in GtGr_dBi_values],
        index=[f"{p} dBm" for p in Pt_dBm_values],
    )
    print(f"\n=== Max Distance Table for Noise Figure F = {F_dB} dB ===")
    print(df.round(4))

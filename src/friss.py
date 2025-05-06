import matplotlib.pyplot as plt
import numpy as np

# === Constants ===
c = 3e8  # Speed of light in m/s
k_B = 1.38e-23  # Boltzmann constant in J/K
T = 290  # Ambient temperature in Kelvin

# === User Input ===
freq_ghz = 4
freq_hz = freq_ghz * 1e9
wavelength = c / freq_hz  # Wavelength in meters

print(f"Calculated wavelength: {wavelength:.3e} m")

# === Adjustable Parameters ===
Pt_dBm = -2
Gt_dBi = 12
Gr_dBi = 12
Gs_dBi = 5.5
S11_dB = -10
Ld_dB = -0.5
B = 1e9
F_dB = 1
SNR_dB = 20
distance_range = np.linspace(0.01, 1, 1000)


# === Utility Functions ===
def db_to_linear(db):
    return 10 ** (db / 10)


def received_power_curve_data(
    freq_ghz=freq_ghz, wavelength=wavelength, distance_range=distance_range
):
    c = 3e8
    k_B = 1.38e-23
    T = 290
    B = 1e9
    Pt_dBm = -2
    Gt_dBi = 12
    Gr_dBi = 12
    Gs_dBi = 5.5
    S11_dB = -10
    Ld_dB = -0.5
    F_dB = 1
    SNR_dB = 20

    def db_to_linear(db):
        return 10 ** (db / 10)

    Pt = db_to_linear(Pt_dBm - 30)
    Gt = db_to_linear(Gt_dBi)
    Gr = db_to_linear(Gr_dBi)
    Gs = db_to_linear(Gs_dBi)
    S11 = db_to_linear(S11_dB)
    Ld = db_to_linear(Ld_dB)
    F = db_to_linear(F_dB)
    SNR = db_to_linear(SNR_dB)

    E = k_B * T
    P_min = E * B * F * SNR
    P_min_dBm = 10 * np.log10(P_min * 1000)

    def received_power(d):
        const_term = Pt * Gt * Gr * Gs**2 * S11 * Ld
        distance_term = (wavelength / (4 * np.pi * d)) ** 4
        return const_term * distance_term

    Pr = received_power(distance_range)
    Pr_dBm = 10 * np.log10(Pr * 1000)

    return distance_range, Pr_dBm, P_min_dBm


def received_power_curve(
    ax, freq_ghz=freq_ghz, wavelength=wavelength, distance_range=distance_range
):
    Pt = db_to_linear(Pt_dBm - 30)  # dBm to Watts
    Gt = db_to_linear(Gt_dBi)
    Gr = db_to_linear(Gr_dBi)
    Gs = db_to_linear(Gs_dBi)
    S11 = db_to_linear(S11_dB)
    Ld = db_to_linear(Ld_dB)
    F = db_to_linear(F_dB)
    SNR = db_to_linear(SNR_dB)

    E = k_B * T
    P_min = E * B * F * SNR
    P_min_dBm = 10 * np.log10(P_min * 1000)

    def received_power(d):
        const_term = Pt * Gt * Gr * Gs**2 * S11 * Ld
        distance_term = (wavelength / (4 * np.pi * d)) ** 4
        return const_term * distance_term

    Pr = received_power(distance_range)
    Pr_dBm = 10 * np.log10(Pr * 1000)

    ax.plot(distance_range, Pr_dBm, label="Received Power")
    ax.axhline(
        y=P_min_dBm,
        color="red",
        linestyle="--",
        label=f"Minimum Detectable Power ({P_min_dBm:.1f} dBm)",
    )

    ax.set_title(
        f"Received Power vs. Distance\n(Resonant Frequency: {freq_ghz} GHz, λ = {wavelength:.2e} m)"
    )
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Power (dBm)")
    ax.set_ylim([-150, 0])
    ax.grid(True)
    ax.legend()


# === Usage ===
fig, ax = plt.subplots(figsize=(8, 6))
received_power_curve(ax, freq_ghz, wavelength, distance_range)
plt.tight_layout()
plt.show()

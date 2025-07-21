import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def plot_temperature_s11():
    """
    Plot S11 vs Frequency for different temperatures.
    Colors: Yellow for low temperature (low dielectric), Red for high temperature (high dielectric)
    Temperature range: 20-170°C with 10°C steps
    """
    # Read the temperature-varying S11 data
    df_temp = pd.read_csv("QwaveTempChange.csv")

    # Get unique dielectric constants (representing different temperatures)
    dielectric_values = sorted(df_temp[df_temp.columns[0]].unique())

    # Temperature range: 20-170°C with 10°C steps
    # Note: Higher dielectric constant corresponds to LOWER temperature
    temperatures = np.linspace(170, 20, len(dielectric_values))

    # Create custom colormap: red to yellow (since high dielectric = low temp = red, low dielectric = high temp = yellow)
    colors = LinearSegmentedColormap.from_list("temp_colors", ["red", "yellow"])(
        np.linspace(0, 1, len(dielectric_values))
    )

    plt.figure(figsize=(12, 8))

    # Lists to store all resonance points
    all_res_freqs = []
    all_res_s11s = []
    all_temps = []

    # Plot S11 for each dielectric constant/temperature
    for i, (dielectric, temp) in enumerate(zip(dielectric_values, temperatures)):
        # Filter data for this dielectric constant
        mask = df_temp[df_temp.columns[0]] == dielectric
        freq_data = df_temp[mask][df_temp.columns[1]]  # Frequency
        s11_data = df_temp[mask][df_temp.columns[2]]  # S11

        plt.plot(
            freq_data,
            s11_data,
            color=colors[i],
            linewidth=1.5,
            alpha=0.8,
            label=f"{temp:.0f}°C (εᵣ={dielectric:.3f})",
        )

        # Find resonance point (minimum S11) for this temperature
        min_idx = s11_data.idxmin()
        res_freq = freq_data.loc[min_idx]
        res_s11 = s11_data.loc[min_idx]

        # Store all resonance points
        all_res_freqs.append(res_freq)
        all_res_s11s.append(res_s11)
        all_temps.append(temp)

    # Get resonance points for lowest and highest temperatures
    low_temp_res_freq = all_res_freqs[0]
    low_temp_res_s11 = all_res_s11s[0]
    low_temp = all_temps[0]

    high_temp_res_freq = all_res_freqs[-1]
    high_temp_res_s11 = all_res_s11s[-1]
    high_temp = all_temps[-1]

    # Add markers for resonance points
    plt.scatter(
        [low_temp_res_freq],
        [low_temp_res_s11],
        color="red",
        edgecolor="black",
        s=100,
        zorder=5,
    )
    plt.scatter(
        [high_temp_res_freq],
        [high_temp_res_s11],
        color="yellow",
        edgecolor="black",
        s=100,
        marker="s",
        zorder=5,
    )

    # Add annotations for resonance frequencies - positioned near the points but offset to avoid blocking
    plt.annotate(
        f"High Temp Resonance\n{low_temp:.0f}°C: {low_temp_res_freq:.3f} GHz\n{low_temp_res_s11:.2f} dB",
        xy=(low_temp_res_freq, low_temp_res_s11),
        xytext=(
            low_temp_res_freq + 0.05,
            low_temp_res_s11 - 8,
        ),  # Offset below and to the right
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1"),
    )

    plt.annotate(
        f"Room Temp Resonance\n{high_temp:.0f}°C: {high_temp_res_freq:.3f} GHz\n{high_temp_res_s11:.2f} dB",
        xy=(high_temp_res_freq, high_temp_res_s11),
        xytext=(
            high_temp_res_freq - 0.40,
            high_temp_res_s11 - 8,
        ),  # Offset further to the left
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1"),
    )

    # Customize the plot
    plt.xlabel("Frequency (GHz)", fontsize=12)
    plt.ylabel("S11 (dB)", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Create a colorbar to show temperature scale
    sm = cm.ScalarMappable(
        cmap=LinearSegmentedColormap.from_list("temp_colors", ["yellow", "red"]),
        norm=Normalize(vmin=20, vmax=170),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label("Temperature (°C)", fontsize=12)

    # Show legend for every 3rd temperature to avoid overcrowding
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles[::3],
        labels[::3],
        bbox_to_anchor=(1.15, 1),
        loc="upper left",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig("temperature_s11_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_temperature_s11()

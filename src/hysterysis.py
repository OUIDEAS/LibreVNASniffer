import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from touchstone import TouchstoneList, Touchstone
from csvList import sensVsDist
from modelplotter import ModelPlotter
import matplotlib.lines as mlines
import os
import re
from datetime import datetime


def plot_hysteresis(tsl: TouchstoneList, timestep):
    # Extract data from the Touchstone object
    temperature_data = np.array(tsl.getTemperatureDataList())
    resonance_frequency = np.array(tsl.getResonanceFrequencyList())
    time_data = np.arange(len(temperature_data))  # Assuming sequential time indices

    # Create the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Resonance Frequency vs Temperature (Hysteresis)
    ax1 = axes[0]
    heating_mask = time_data <= timestep
    cooling_mask = time_data > timestep

    # Line plot for heating and cooling cycles
    ax1.plot(
        temperature_data[heating_mask],
        resonance_frequency[heating_mask],
        color="red",
        label="Heating Cycle",
    )
    ax1.plot(
        temperature_data[cooling_mask],
        resonance_frequency[cooling_mask],
        color="blue",
        label="Cooling Cycle",
    )

    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("Resonant Frequency (Hz)")
    # ax1.set_title("Resonance Frequency vs Temperature")
    ax1.legend()
    ax1.grid(True)

    # Temperature vs Time
    ax2 = axes[1]
    ax2.plot(
        time_data[heating_mask],
        temperature_data[heating_mask],
        color="red",
        label="Heating Cycle",
    )
    ax2.plot(
        time_data[cooling_mask],
        temperature_data[cooling_mask],
        color="blue",
        label="Cooling Cycle",
    )

    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Temperature (°C)")
    # ax2.set_title("Temperature vs Time")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("hysteresisPlot.png")
    plt.show()


# Example usage
touchstone_list = TouchstoneList.loadTouchstoneListFromCSV(
    "./data/run-20250613_195252/csv_20250613_195252.csv"
)
plot_hysteresis(touchstone_list, timestep=32)

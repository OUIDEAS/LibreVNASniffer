import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Function to read CSV and return frequency & S11 values


def read_s11_data(file_path):
    df = pd.read_csv(file_path)
    freq = df.iloc[:, 0]  # First column is frequency
    s11 = df.iloc[:, 1]  # Second column is S11 (dB)
    return freq, s11


# Function to find the minimum S11 value and its corresponding frequency


def find_min_s11(freq, s11):
    min_index = s11.idxmin()
    return freq[min_index], s11[min_index]


# File paths for the two CSV files
root = "./src/s11hfss/"  # Since we're in the s11hfss directory
file1 = root + "CoaxialS11.csv"  # Replace with actual path
file2 = root + "QwaveS11.csv"  # Replace with actual path

# Read the two datasets
freq1, s11_1 = read_s11_data(file1)
freq2, s11_2 = read_s11_data(file2)

# Find the minimum points
min_freq1, min_s11_1 = find_min_s11(freq1, s11_1)
min_freq2, min_s11_2 = find_min_s11(freq2, s11_2)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(freq1, s11_1, label="Coaxial Feedline", color="blue", linestyle="-")
plt.plot(
    freq2, s11_2, label="Quarter Wave Transform Feedline", color="red", linestyle="-"
)

# Mark the minimum points
plt.scatter(min_freq1, min_s11_1, color="blue", marker="o")
plt.scatter(min_freq2, min_s11_2, color="red", marker="s")

# Annotate the min points with frequency and magnitude
plt.text(
    min_freq1,
    min_s11_1,
    f"({min_freq1:.3f} GHz, {min_s11_1:.2f} dB)",
    fontsize=10,
    verticalalignment="bottom",
    horizontalalignment="right",
    color="blue",
)
plt.text(
    min_freq2,
    min_s11_2,
    f"({min_freq2:.3f} GHz, {min_s11_2:.2f} dB)",
    fontsize=10,
    verticalalignment="bottom",
    horizontalalignment="left",
    color="red",
)

# Labels and title
plt.xlabel("Frequency (GHz)")
plt.ylabel("S11 (dB)")
plt.legend()
plt.grid()

# Show the plot
plt.show()

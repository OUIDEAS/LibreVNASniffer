import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from touchstone import TouchstoneList, Touchstone
from csvList import sensVsDist
from modelplotter import ModelPlotter
import matplotlib.lines as mlines
import os
import re
import matplotlib.lines as mlines


def format_date_with_slashes(date_str):
    """
    Converts a date string from YYYYMMDD to YYYY/MM/DD format.

    Args:
        date_str (str): The date string in YYYYMMDD format.

    Returns:
        str: The formatted date string in YYYY/MM/DD format.
    """
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError("Input must be a string in YYYYMMDD format")

    return f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"


def find_csv_files(root_dir):
    csv_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                csv_files.append(os.path.join(dirpath, filename))
    return csv_files


# Define a blacklist of paths
blacklist = [
    "data/run-20240701_140555/csv_20240701_140555.csv",
    "data/run-20240702_131645/csv_20240702_131645.csv",
]

# Usage
csv_list = find_csv_files("data")

# Remove blacklisted paths
csv_list = [path for path in csv_list if path not in blacklist]


# Sort csv_list from oldest to newest based on the timestamp in the file path
def extract_timestamp(path):
    match = re.search(r"(\d{8}_\d{6})", path)
    return match.group(1) if match else ""


csv_list = sorted(csv_list, key=extract_timestamp)

print("Sorted CSV files:")
print(csv_list)

print("CSV files found:")
print(csv_list)
print("Length of csv_list:", len(csv_list))

# input("Press Enter to continue.")
print("loading ", len(sensVsDist), " datasets")
datasets = [TouchstoneList.loadTouchstoneListFromCSV(path) for path in csv_list]
filtered_datasets = []

for ts in datasets:
    print(ts.name)

    if len(ts.touchstones) < 5:
        print("Dataset removed due to insufficient touchstones")
        continue

    if ts.getR2() < 0.2:
        print("Dataset removed due to low R² score")
        continue

    filtered_datasets.append(ts)

datasets = filtered_datasets  # Replace original with filtered


touchstoneTuples = [
    (
        ts.name if ts.name else "No name",
        ts.getRootFrequency() * 1e-9,
        abs(ts.getSlopeAndInterceptOfResonantFreq()[0]) * 1e-6,
        ts.getR2(),
        ts.config.data["distance"] if ts.config and ts.config.data else 0,
    )
    for ts in datasets
]

print(*touchstoneTuples, sep="\n")

# wait for a second


# Create a DataFrame from the list of tuples
df = pd.DataFrame(
    touchstoneTuples, columns=["Name", "Root", "Sensitivity", "R2", "Distance"]
)

df["invSensitivity"] = 1 / df["Sensitivity"]


wait = input("Press Enter to continue.")
center1, center2 = 3.1, 3.7


# Define the bins for the specific ranges you want
bins = [0, 1, 201, 351, 501, 751, 901]

# Group by the defined distance intervals
df["Distance Group"] = pd.cut(df["Distance"], bins=bins, right=False)

# Convert the 'Distance Group' to a categorical column with ordered categories
df["Distance Group"] = pd.Categorical(
    df["Distance Group"],
    categories=pd.cut(bins, bins=bins, right=False).categories,
    ordered=True,
)

counts = [
    len(df[df["Distance Group"] == group])
    for group in df["Distance Group"].cat.categories
]
group_labels = ["NA", "200", "350", "500", "750", "900"]
labels = [f"{group}\n(n={count})" for group, count in zip(group_labels, counts)]


# Use the existing Distance Group column for coloring
distance_colors = {
    pd.Interval(0, 1, closed="left"): "gray",  # Probably for missing/edge cases
    pd.Interval(1, 201, closed="left"): "green",
    pd.Interval(201, 351, closed="left"): "blue",
    pd.Interval(351, 501, closed="left"): "orange",
    pd.Interval(501, 751, closed="left"): "purple",
    pd.Interval(751, 901, closed="left"): "red",
}
custom_labels = {
    pd.Interval(0, 1, closed="left"): "NA",
    pd.Interval(1, 201, closed="left"): "200",
    pd.Interval(201, 351, closed="left"): "350",
    pd.Interval(351, 501, closed="left"): "500",
    pd.Interval(501, 751, closed="left"): "750",
    pd.Interval(751, 901, closed="left"): "900",
}


# Map colors to the Distance Group
print(df["Distance Group"].unique())

df["Bar Color"] = df["Distance Group"].apply(
    lambda group: distance_colors.get(group, "black")
)


# Plot the bar graph for R² scores
plt.figure(figsize=(10, 6))
plt.bar(
    range(1, len(df["R2"]) + 1),
    df["R2"],
    color=df["Bar Color"],
)
plt.xlabel("Dataset Index")
plt.ylabel("R² Score")

# Extract dates for the earliest and latest tests
earliest_date = format_date_with_slashes(
    extract_timestamp(csv_list[0])[:8]
)  # First 8 characters for YYYYMMDD
latest_date = format_date_with_slashes(
    extract_timestamp(csv_list[-1])[:8]
)  # First 8 characters for YYYYMMDD
plt.title(
    f"R² Scores of Touchstone Datasets\n(Earliest: {earliest_date}, Latest: {latest_date})"
)

# Add legend
legend_handles = [
    mlines.Line2D(
        [], [], color=distance_colors[interval], label=custom_labels[interval]
    )
    for interval in df["Distance Group"].cat.categories
    if interval in custom_labels  # just in case some groups are empty
]

plt.legend(handles=legend_handles, title="Distance Categories")

plt.xticks(range(1, len(df["R2"]) + 1))
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

df["Cluster"] = df["Root"].apply(
    lambda x: "Cluster1" if abs(x - center1) < abs(x - center2) else "Cluster2"
)
# Split into two DataFrames
df_cluster1 = df[df["Cluster"] == "Cluster1"].drop(columns=["Cluster"])
df_cluster2 = df[df["Cluster"] == "Cluster2"].drop(columns=["Cluster"])
# PLOTTING
# Create a figure for each plot
sensVsDistPlot, sensVsDistAx = plt.subplots(figsize=(13 / 2, 8 / 2))
accVsDistPlot, accVsDistAx = plt.subplots(figsize=(13 / 2, 8 / 2))
freqVsDistPlot, freqVsDistAx = plt.subplots(figsize=(13 / 2, 8 / 2))
# First subplot: Boxplot of Sensitivity
sensVsDistAx.boxplot(
    [
        df_cluster2[df_cluster2["Distance Group"] == group]["Sensitivity"]
        for group in df_cluster2["Distance Group"].cat.categories
    ],
    labels=labels,
)
# sensVsDistAx.set_title("Boxplot of Sensitivity by Distance Group")
sensVsDistAx.set_xlabel("Distance (mm) Group")
sensVsDistAx.set_ylabel("Sensitivity (MHz/°C)")
sensVsDistAx.grid(True)

# Add individual points
additional_points_x = [2, 3, 4]  # Corresponding x positions for "350", "500", "750"
additional_points_y = [0.027, 0.056, 0.059]
colors = ["blue", "orange", "red"]
labels2 = ["N-35", "N-50", "N-75"]
sensVsDistAx.scatter(
    additional_points_x,
    additional_points_y,
    color=colors,
    marker="o",
)
# Create legend handles manually
legend_handles = [
    mlines.Line2D(
        [], [], color=color, marker="o", linestyle="None", markersize=8, label=label
    )
    for color, label in zip(colors, labels2)
]

# Add legend with correct colors
sensVsDistAx.legend(handles=legend_handles)

# Second subplot: Boxplot of Accuracy
accVsDistAx.boxplot(
    [
        df_cluster2[df_cluster2["Distance Group"] == group]["R2"]
        for group in df_cluster2["Distance Group"].cat.categories
    ],
    labels=labels,
)
# accVsDistAx.set_title("Boxplot of Accuracy by Distance Group")
accVsDistAx.set_xlabel("Distance (mm) Group")
accVsDistAx.set_ylabel("Accuracy (R²)")
accVsDistAx.grid(True)

# Third subplot: Scatter plot
unique_distances = sorted(df["Distance"].unique(), reverse=True)
colors = ["red", "blue", "green", "purple", "orange", "yellow"]
for i, distance in enumerate(unique_distances):
    subset = df[df["Distance"] == distance]
    freqVsDistAx.scatter(
        subset["Root"],
        subset["Sensitivity"],
        color=colors[i % len(colors)],
        label=f"Distance {distance} mm",
        edgecolors="black",
    )
# freqVsDistAx.set_title("Root Frequency vs Sensitivity")
freqVsDistAx.set_xlabel("Root Frequency (MHz)")
freqVsDistAx.set_ylabel("Sensitivity (MHz/°C)")
freqVsDistAx.legend(title="Distance (mm)")
freqVsDistAx.grid(True)


plt.tight_layout()
ModelPlotter.saveFigure(sensVsDistPlot, "sensVsDistPlot")
ModelPlotter.saveFigure(accVsDistPlot, "accVsDistPlot")
ModelPlotter.saveFigure(freqVsDistPlot, "freqVsDistPlot")

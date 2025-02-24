import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from touchstone import TouchstoneList, Touchstone
from csvList import sensVsDist
from modelplotter import ModelPlotter
import matplotlib.lines as mlines


print("loading ", len(sensVsDist), " datasets")
datasets = [TouchstoneList.loadTouchstoneListFromCSV(path) for path in sensVsDist]
touchstoneTuples = [
    (
        ts.name if ts.name else "No name",
        ts.getRootFrequency() * 1e-9,
        abs(ts.getSlopeAndInterceptOfResonantFreq()[0]) * 1e-6,
        ts.getR2(),
        ts.config.data["distance"],
    )
    for ts in datasets
]

print(*touchstoneTuples, sep="\n")

# wait for a second
# wait = input("Press Enter to continue.")


# Create a DataFrame from the list of tuples
df = pd.DataFrame(
    touchstoneTuples, columns=["Name", "Root", "Sensitivity", "R2", "Distance"]
)

df["invSensitivity"] = 1 / df["Sensitivity"]

center1, center2 = 3.1, 3.7
df["Cluster"] = df["Root"].apply(
    lambda x: "Cluster1" if abs(x - center1) < abs(x - center2) else "Cluster2"
)
# Split into two DataFrames
df_cluster1 = df[df["Cluster"] == "Cluster1"].drop(columns=["Cluster"])
df_cluster2 = df[df["Cluster"] == "Cluster2"].drop(columns=["Cluster"])

# Define the bins for the specific ranges you want
bins = [0, 201, 351, 501, 751, 901]

# Group by the defined distance intervals
df_cluster2["Distance Group"] = pd.cut(df_cluster2["Distance"], bins=bins, right=False)

# Convert the 'Distance Group' to a categorical column with ordered categories
df_cluster2["Distance Group"] = pd.Categorical(
    df_cluster2["Distance Group"],
    categories=pd.cut(bins, bins=bins, right=False).categories,
    ordered=True,
)

counts = [
    len(df_cluster2[df_cluster2["Distance Group"] == group])
    for group in df_cluster2["Distance Group"].cat.categories
]
group_labels = ["200", "350", "500", "750", "900"]
labels = [f"{group}\n(n={count})" for group, count in zip(group_labels, counts)]

# Create a figure for each plot
sensVsDistPlot, sensVsDistAx = plt.subplots(figsize=(13, 8))
accVsDistPlot, accVsDistAx = plt.subplots(figsize=(13, 8))
freqVsDistPlot, freqVsDistAx = plt.subplots(figsize=(13, 8))
# First subplot: Boxplot of Sensitivity
sensVsDistAx.boxplot(
    [
        df_cluster2[df_cluster2["Distance Group"] == group]["Sensitivity"]
        for group in df_cluster2["Distance Group"].cat.categories
    ],
    labels=labels,
)
sensVsDistAx.set_title("Boxplot of Sensitivity by Distance Group")
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
accVsDistAx.set_title("Boxplot of Accuracy by Distance Group")
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
freqVsDistAx.set_title("Root Frequency vs Sensitivity")
freqVsDistAx.set_xlabel("Root Frequency (MHz)")
freqVsDistAx.set_ylabel("Sensitivity (MHz/°C)")
freqVsDistAx.legend(title="Distance (mm)")
freqVsDistAx.grid(True)


plt.tight_layout()
ModelPlotter.saveFigure(sensVsDistPlot, "sensVsDistPlot")
ModelPlotter.saveFigure(accVsDistPlot, "accVsDistPlot")
ModelPlotter.saveFigure(freqVsDistPlot, "freqVsDistPlot")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from touchstone import TouchstoneList, Touchstone
from csvList import sensVsDist


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
wait = input("Press Enter to continue.")


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

# Create a 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# First subplot: Boxplot of Sensitivity
axes[0, 0].boxplot(
    [
        df_cluster2[df_cluster2["Distance Group"] == group]["Sensitivity"]
        for group in df_cluster2["Distance Group"].cat.categories
    ],
    labels=labels,
)
axes[0, 0].set_title("Boxplot of Sensitivity by Distance Group")
axes[0, 0].set_xlabel("Distance (mm) Group")
axes[0, 0].set_ylabel("Sensitivity (MHz/°C)")
axes[0, 0].grid(True)

# Add individual points
additional_points_x = [2, 3, 4]  # Corresponding x positions for "350", "500", "750"
additional_points_y = [0.027, 0.056, 0.059]
axes[0, 0].scatter(
    additional_points_x,
    additional_points_y,
    color=["blue", "orange", "red"],
    marker="o",
)
axes[0, 0].legend(["N-35", "N-50", "N-75"])

# Second subplot: Boxplot of Accuracy
axes[0, 1].boxplot(
    [
        df_cluster2[df_cluster2["Distance Group"] == group]["R2"]
        for group in df_cluster2["Distance Group"].cat.categories
    ],
    labels=labels,
)
axes[0, 1].set_title("Boxplot of Accuracy by Distance Group")
axes[0, 1].set_xlabel("Distance (mm) Group")
axes[0, 1].set_ylabel("Accuracy (R²)")
axes[0, 1].grid(True)

# Third subplot: Scatter plot
unique_distances = sorted(df["Distance"].unique(), reverse=True)
colors = ["red", "blue", "green", "purple", "orange", "yellow"]
for i, distance in enumerate(unique_distances):
    subset = df[df["Distance"] == distance]
    axes[1, 0].scatter(
        subset["Root"],
        subset["Sensitivity"],
        color=colors[i % len(colors)],
        label=f"Distance {distance} mm",
        edgecolors="black",
    )
axes[1, 0].set_title("Root Frequency vs Sensitivity")
axes[1, 0].set_xlabel("Root Frequency (MHz)")
axes[1, 0].set_ylabel("Sensitivity (MHz/°C)")
axes[1, 0].legend(title="Distance (mm)")
axes[1, 0].grid(True)

# Hide the fourth subplot (empty for now)
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()

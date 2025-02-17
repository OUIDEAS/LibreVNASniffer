import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from touchstone import TouchstoneList, Touchstone
from csvList import sensVsDist

data = [
    (200, 0.022),
    (200, 0.066),
    (300, 0.052),
    (400, 0.023),
    (250, 0.061),
    (370, 0.023),
    (450, 0.028),
    (310, 0.030),
    (340, 0.014),
    (230, 0.108),
    (230, 0.083),
    (320, 0.014),
    (200, 0.044),
    (200, 0.041),
]
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

print(df_cluster1)
print(df_cluster2)

# Define the bins for the specific ranges you want
bins = [0, 400, 600, 800]

# Group by the defined distance intervals
df_cluster2["Distance Group"] = pd.cut(df_cluster2["Distance"], bins=bins, right=False)

# Convert the 'Distance Group' to a categorical column with ordered categories
df_cluster2["Distance Group"] = pd.Categorical(
    df_cluster2["Distance Group"],
    categories=pd.cut(bins, bins=bins, right=False).categories,
    ordered=True,
)

# Create the boxplot
plt.figure(figsize=(10, 6))
plt.boxplot(
    [
        df_cluster2[df_cluster2["Distance Group"] == group]["invSensitivity"]
        for group in df_cluster2["Distance Group"].cat.categories
    ],
    labels=["350", "500", "750"],
)


# Add individual points at specific distances
additional_points_x = [1, 2, 3]  # Corresponding x positions for "350", "500", "750"
additional_points_y = [0.027, 0.056, 0.059]
additional_points_y = [1 / y for y in additional_points_y]

plt.scatter(
    additional_points_x[0],
    additional_points_y[0],
    color="blue",
    marker="o",
    label="N-35",
)
plt.scatter(
    additional_points_x[1],
    additional_points_y[1],
    color="orange",
    marker="o",
    label="N-50",
)
plt.scatter(
    additional_points_x[2],
    additional_points_y[2],
    color="red",
    marker="o",
    label="N-75",
)

plt.title("Boxplot of Sensitivity by Distance Group")
plt.xlabel("Distance (mm) Group")
plt.ylabel("Sensitivity (°C/MHz)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()


# Create a figure
plt.figure(figsize=(10, 6))

# Get unique distances and assign colors
unique_distances = sorted(df["Distance"].unique(), reverse=True)
print(unique_distances)

colors = ["red", "blue", "green", "purple", "orange"]  # Add more if needed

# Plot each distance separately
for i, distance in enumerate(unique_distances):
    subset = df[df["Distance"] == distance]
    plt.scatter(
        subset["Root"],
        subset["Sensitivity"],
        color=colors[i % len(colors)],  # Cycle through colors if needed
        label=f"Distance {distance} mm",
        edgecolors="black",
    )

# Labels and legend
plt.ylabel("Sensitivity (MHz/°C)")
plt.xlabel("Root Frequency (MHz)")
plt.title("Scatter Plot of Root Frequency vs Sensitivity by Distance")
plt.legend(title="Distance (mm)")
plt.grid(True)

# Show plot
plt.show()

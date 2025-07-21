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


def format_date_with_slashes(date_str):
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


def extract_timestamp(path):
    match = re.search(r"(\d{8}_\d{6})", path)
    return match.group(1) if match else ""


def plot_r2_scores(df, csv_list):
    earliest_date = format_date_with_slashes(extract_timestamp(csv_list[0])[:8])
    latest_date = format_date_with_slashes(extract_timestamp(csv_list[-1])[:8])

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(df["R2"]) + 1), df["R2"], color=df["Distance Group Color"])
    plt.xlabel("Dataset Index")
    plt.ylabel("R² Score")
    plt.title(
        f"R² Scores of Touchstone Datasets\n(Earliest: {earliest_date}, Latest: {latest_date})"
    )

    # Create legend using unique Distance Group Names and Colors
    unique_groups = df[
        ["Distance Group Name", "Distance Group Color"]
    ].drop_duplicates()
    legend_handles = [
        mlines.Line2D(
            [], [], color=row["Distance Group Color"], label=row["Distance Group Name"]
        )
        for _, row in unique_groups.iterrows()
    ]
    plt.legend(handles=legend_handles, title="Distance Categories")
    plt.xticks(range(1, len(df["R2"]) + 1))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    ModelPlotter.saveFigure(plt.gcf(), "r2ScoresPlot")


def plot_sensitivity_by_distance(df_cluster2, labels):
    """
    Plots sensitivity by distance group and overlays received power curve data.
    This function generates a boxplot of sensitivity values grouped by distance
    categories, along with scatter points for additional data. It also overlays
    a received power curve on a secondary y-axis, showing the relationship between
    distance and received power in dBm.
    Args:
        df_cluster2 (pd.DataFrame): DataFrame containing sensitivity and distance group data.
            Expected columns:
                - "Distance Group": Categorical data representing distance bins.
                - "Sensitivity": Sensitivity values (MHz/°C).
        labels (list): Labels for the scatter points corresponding to additional data.
    Returns:
        None: The function saves the generated plot as a file using ModelPlotter.saveFigure.
    Notes:
        - The boxplot represents sensitivity grouped by predefined distance bins.
        - Additional scatter points are plotted for specific distances and sensitivities.
        - A secondary y-axis displays the received power curve data, including a horizontal
          line for the minimum detectable power.
        - The function uses `received_power_curve_data` from the `friss` module to obtain
          distances and power values.
    Example:
        >>> plot_sensitivity_by_distance(df_cluster2, labels=["N-35", "N-50", "N-75"])
    """
    from friss import received_power_curve_data

    fig, ax1 = plt.subplots(figsize=(8, 5))
    # Define midpoints of each bin (in mm)
    group_midpoints = [0.5, 200, 350, 500, 750, 900]  # Rough center of each bin
    group_labels = ["0", "200", "350", "500", "750", "900"]
    # Boxplot for sensitivity by distance group
    groups = df_cluster2["Distance Group"].cat.categories
    data = [
        df_cluster2[df_cluster2["Distance Group"] == group]["Sensitivity"]
        for group in groups
    ]
    ax1.boxplot(data, positions=group_midpoints, widths=90)  # widths can be adjusted
    # X axis label locations and text
    ax1.set_xticks(group_midpoints)
    counts = [len(d) for d in data]
    ax1.set_xticklabels([f"{label}" for label, count in zip(group_labels, counts)])

    ax1.set_xlabel("Distance (mm)")
    ax1.set_ylabel("Sensitivity (MHz/°C)")
    ax1.grid(True)

    # Add individual scatter points
    additional_points_x = [350, 500, 750]
    additional_points_y = [0.027, 0.056, 0.059]
    colors = ["blue", "orange", "red"]
    labels2 = ["N-35", "N-50", "N-75"]
    ax1.scatter(additional_points_x, additional_points_y, color=colors)

    legend_handles = [
        mlines.Line2D(
            [], [], color=color, marker="o", linestyle="None", markersize=8, label=label
        )
        for color, label in zip(colors, labels2)
    ]

    # === Twin y-axis for received power ===
    ax2 = ax1.twinx()

    # distances_m is already in mm!
    distances_mm, powers_dBm, P_min_dBm = received_power_curve_data()
    distances_mm = np.array(distances_mm) * 1000  # Convert to mm
    # remove distances less than 200mm, remove associated powers
    powers_dBm = np.array(powers_dBm)
    # Remove distances less than 200mm and their associated powers
    # mask = distances_mm >= 200  # Create a boolean mask for distances >= 200mm
    # distances_mm = distances_mm[mask]  # Filter distances_mm
    # powers_dBm = powers_dBm[mask]  # Filter associated powers_dBm

    print(distances_mm)

    ax2.plot(distances_mm, powers_dBm, color="blue", label="Received Power")
    ax2.axhline(
        P_min_dBm,
        color="red",
        linestyle="--",
        label=f"Min Detectable ({P_min_dBm:.1f} dBm)",
    )
    ax2.set_ylabel("Received Power (dBm)")
    # Set power curve axis to appear 10 dB higher
    ymin = -90
    ymax = -30
    offset = 15
    ax2.set_ylim(ymin + offset, ymax + offset)

    # Legend combining both
    combined_handles = legend_handles + [
        mlines.Line2D([], [], color="blue", label="Received Power"),
        mlines.Line2D(
            [], [], color="red", linestyle="--", label="Min Detectable Power"
        ),
    ]
    ax2.legend(handles=combined_handles, loc="upper right")

    # plt.title(f"Received Power vs. Distance\n(Resonant Frequency: 4 GHz,   ")
    plt.tight_layout()
    ModelPlotter.saveFigure(fig, "sensVsDistPlot")


def plot_accuracy_by_distance(df_cluster2, labels):
    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Exclude the group from 0 to 1
    filtered_groups = [
        group
        for group in df_cluster2["Distance Group"].cat.categories
        if group != pd.Interval(0, 1, closed="left")
    ]
    filtered_labels = [
        label
        for group, label in zip(df_cluster2["Distance Group"].cat.categories, labels)
        if group != pd.Interval(0, 1, closed="left")
    ]

    ax.boxplot(
        [
            df_cluster2[df_cluster2["Distance Group"] == group]["R2"]
            for group in filtered_groups
        ],
        labels=filtered_labels,
    )
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Accuracy (R²)")
    ax.grid(True)
    plt.tight_layout()
    ModelPlotter.saveFigure(fig, "accVsDistPlot")


def plot_root_vs_sensitivity(df):
    fig = plt.figure(figsize=(6.5, 4))

    # Sort the DataFrame by Distance Group to ensure proper order
    sorted_groups = (
        df[["Distance Group", "Distance Group Name", "Distance Group Color"]]
        .drop_duplicates()
        .sort_values(by="Distance Group")
    )

    # Iterate over each unique Distance Group Name and plot its data
    for group_name, group_color in sorted_groups[
        ["Distance Group Name", "Distance Group Color"]
    ].values:
        group_data = df[df["Distance Group Name"] == group_name]
        plt.scatter(
            group_data["Root"],
            group_data["Sensitivity"],
            color=group_color,
            label=group_name,
        )

    plt.xlabel("Initial Resonant Frequency (GHz)")
    plt.ylabel("Sensitivity (MHz/°C)")
    plt.legend(title="Distance Groups (mm)")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    ModelPlotter.saveFigure(fig, "freqVsDistPlot")


# === Main script ===
if __name__ == "__main__":
    blacklist = [
        "data/run-20240701_140555/csv_20240701_140555.csv",
        "data/run-20240702_131645/csv_20240702_131645.csv",
    ]

    csv_list = sorted(
        [path for path in find_csv_files("data") if path not in blacklist],
        key=extract_timestamp,
    )
    csv_list = sensVsDist

    datasets = [
        ts
        for ts in [TouchstoneList.loadTouchstoneListFromCSV(path) for path in csv_list]
        if len(ts.touchstones) >= 5 and ts.getR2() >= 0.2
    ]
    print(len(datasets), "datasets loaded")

    touchstoneTuples = [
        (
            ts.name if ts.name else "No name",
            ts.getRootFrequency() * 1e-9,
            abs(ts.getSlopeAndInterceptOfResonantFreq()[0]) * 1e-6,
            ts.getR2(),
            ts.config.data["distance"] if ts.config and ts.config.data else 0,
            ts.getTemperatureDataList(),
        )
        for ts in datasets
    ]

    df = pd.DataFrame(
        touchstoneTuples,
        columns=["Name", "Root", "Sensitivity", "R2", "Distance", "Temperature"],
    )
    df["invSensitivity"] = 1 / df["Sensitivity"]

    bins = [0, 1, 201, 351, 501, 751, 901]
    df["Distance Group"] = pd.cut(df["Distance"], bins=bins, right=False)
    df["Distance Group"] = pd.Categorical(
        df["Distance Group"],
        categories=pd.cut(bins, bins=bins, right=False).categories,
        ordered=True,
    )

    group_labels = ["NA", "200", "350", "500", "750", "900"]
    counts = [
        len(df[df["Distance Group"] == group])
        for group in df["Distance Group"].cat.categories
    ]
    labels = [f"{group}" for group, count in zip(group_labels, counts)]

    distance_colors = {
        pd.Interval(0, 1, closed="left"): "gray",
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

    # Add new columns to the DataFrame
    df["Distance Group Name"] = df["Distance Group"].apply(
        lambda group: custom_labels.get(group, "Unknown")
    )
    df["Distance Group Color"] = df["Distance Group"].apply(
        lambda group: distance_colors.get(group, "gray")
    )

    df["Bar Color"] = df["Distance Group"].apply(
        lambda group: distance_colors.get(group, "black")
    )

    center1, center2 = 3.1, 3.7
    df["Cluster"] = df["Root"].apply(
        lambda x: "Cluster1" if abs(x - center1) < abs(x - center2) else "Cluster2"
    )
    df_cluster2 = df[df["Cluster"] == "Cluster2"].drop(columns=["Cluster"])
    # Analyze and print touchstone data
    rising_count = {}
    falling_count = {}

    for _, row in df.iterrows():
        # Determine if the temperature is rising or falling
        temperature_data = row["Temperature"]
        if temperature_data[-1] > temperature_data[0]:
            trend = "Rising"
            rising_count[row["Distance Group"]] = (
                rising_count.get(row["Distance Group"], 0) + 1
            )
        else:
            trend = "Falling"
            falling_count[row["Distance Group"]] = (
                falling_count.get(row["Distance Group"], 0) + 1
            )

        # Print touchstone details
        print(
            f"Touchstone: {row['Name']}, Distance: {row['Distance']} mm, Temperature: {trend}"
        )

    # Print summary
    print("\nSummary:")
    for group in df["Distance Group"].cat.categories:
        rising = rising_count.get(group, 0)
        falling = falling_count.get(group, 0)
        print(f"Distance Group {group}: Rising = {rising}, Falling = {falling}")

    plot_r2_scores(df, csv_list)
    plot_sensitivity_by_distance(df_cluster2, labels)
    plot_accuracy_by_distance(df_cluster2, labels)
    plot_root_vs_sensitivity(df_cluster2)

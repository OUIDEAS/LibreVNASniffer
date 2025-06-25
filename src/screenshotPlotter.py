import matplotlib.pyplot as plt
import numpy as np
from touchstone import Touchstone
import os
import matplotlib.patches as mpatches

# A list of Touchstone file pairs, each pair is a time domain reading with and without the sensor. Distances vary between pairs
timeDomianDataPairs = [
    (
        "./screenshots/screenshot-20250318_124821/screenshot_20250318_124821.csv",
        "./screenshots/screenshot-20250318_124511/screenshot_20250318_124511.csv",
    ),
    (
        "./screenshots/screenshot-20250318_141012/screenshot_20250318_141012.csv",
        "./screenshots/screenshot-20250318_141047/screenshot_20250318_141047.csv",
    ),
    (
        "./screenshots/screenshot-20250318_142201/screenshot_20250318_142201.csv",
        "./screenshots/screenshot-20250318_142228/screenshot_20250318_142228.csv",
    ),
]

for pair in timeDomianDataPairs:
    # Plot a figure limited from 0 to 20 ns
    # two plots ontop of each other
    # one is with sensor, one is without
    # x axis is time in ns
    # y axis is magnitude in dB

    # Load the data
    withSensor = Touchstone.loadSingletonTouchstoneFromCSV(pair[0])
    withoutSensor = Touchstone.loadSingletonTouchstoneFromCSV(pair[1])

    # Plot the data
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(
        np.array(withSensor.getFrequencyRange()) * 1e9,
        withSensor.getDataMagnitude(),
        label="S21 Time Domain",
        color="blue",
    )
    # ax.plot(
    #     np.array(withoutSensor.getFrequencyRange()) * 1e9,
    #     withoutSensor.getDataMagnitude(),
    #     label="Without Sensor",
    #     color="red",
    # )

    # Highlight range from 4.5 ns to 8 ns
    highlight = ax.axvspan(4, 6, color="yellow", alpha=0.3)
    highlight_crosstalk = ax.axvspan(2.5, 4, color="gray", alpha=0.3)

    # Create a legend entry for the shaded region
    highlight_patch = mpatches.Patch(
        color="yellow", alpha=0.3, label="Time Gated Window"
    )
    highlight_patch_crosstalk = mpatches.Patch(
        color="gray", alpha=0.3, label="Interrogator Antenna Crosstalk"
    )

    # Add the legend
    ax.legend(
        handles=[highlight_patch, highlight_patch_crosstalk]
        + ax.get_legend_handles_labels()[0]
    )

    ax.set_xlabel("Seconds (ns)")
    ax.set_xlim(0, 20)  # Limit to 20 ns
    ax.set_ylabel("Magnitude (dB)")
    ax.set_ylim(-80, -25)
    ax.grid()

    fig.show()
    print("Press enter to continue")
    input()

# print("Data: ", data)
# print("Data Length: ", len(data))
# print("Data Shape: ", np.array(data).shape)
# print(touchstone.getResonanceFrequency()[0])
# # Plot freq vs mag
# fig, ax = plt.subplots()
# print(touchstone.getFrequencyRange())
# nanoseconds = np.array(touchstone.getFrequencyRange()) * 1e9
# print(nanoseconds)

# ax.plot(nanoseconds, touchstone.getDataMagnitude())
# ax.set_title("Resonance Frequency")
# ax.set_xlabel("Seconds (ns)")
# ax.set_ylabel("Magnitude (dB)")
# ax.grid()
# fig.show()

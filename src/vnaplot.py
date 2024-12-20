#!/usr/bin/env python3

from librevna import libreVNA
from vnacommandcenter import VNACommandCenter
import numpy as np
import matplotlib.pyplot as plt
from touchstone import Touchstone as touch
from touchstone import TouchstoneList
from matplotlib.animation import FuncAnimation
from thermocouple import Thermocouple
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from dataconfig import DataConfig
import pandas as pd
from sklearn.linear_model import LinearRegression

COMPLEX_LINES = 10


class VNAPlot:
    def __init__(
        self,
        tsList: TouchstoneList,
        config: DataConfig,
    ):
        # Config
        np.set_printoptions(precision=10)
        self.markers = []
        self.config = config
        self.tsList = tsList

        # Initialize plot
        self.fig, axs = plt.subplots(3, 4, dpi=90)
        self.ax1 = axs[0, 0]
        self.ax2 = axs[1, 0]
        self.ax3 = axs[2, 0]
        self.ax4 = axs[0, 1]
        self.ax5 = axs[1, 1]
        self.ax6 = axs[2, 1]
        self.ax7 = axs[0, 2]
        self.ax8 = axs[1, 2]
        self.ax9 = axs[2, 2]
        self.axStats = axs[0, 3]
        self.axStats.axis("off")
        self.axMultiLine = axs[1, 3]
        self.stats = {}
        self.addStat("Distance", config.data["distance"], 1, "mm")

        self.fig.set_size_inches(18, 10)
        (self.line1,) = self.ax1.plot([], [], "r-", linewidth=1)
        (self.line2,) = self.ax2.plot([], [], linewidth=1, color="blue")
        (self.line3,) = self.ax3.plot([], [], linewidth=1, color="green")
        (self.line5,) = self.ax5.plot([], [], linewidth=1, color="brown")
        (self.line6,) = self.ax6.plot([], [], linewidth=1, color="purple")
        (self.line7,) = self.ax7.plot([], [], linewidth=1, color="blue")
        (self.line8,) = self.ax8.plot([], [], linewidth=1, color="purple")
        (self.line9,) = self.ax9.plot([], [], linewidth=1, color="purple")

        # Generate sample data
        self.num_points_time = 120  # Two minutes of data
        self.num_points_frequency = 50
        data = (
            np.random.rand(self.num_points_time * 10, self.num_points_frequency) * 50
        ) - 70
        # Print information about the data
        # print("Data shape:", data.shape)
        # print("Data type:", data.dtype)
        # print("Data range:", np.min(data), np.max(data))
        self.fig.colorbar(
            self.ax4.imshow(
                data,
                cmap="viridis",
                vmin=config.data["minDB"],
                vmax=config.data["maxDB"],
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                extent=[0, config.data["points"], 0, self.num_points_time],
            ),
            ax=self.ax4,
        )

        self.ax4.set_label("Magnitude (dB)")

        # Plot 1: Mag Buffer
        self.ax1.set_xlim(config.data["freqStart"] * 1e9, config.data["freqEnd"] * 1e9)
        # Adjust limits based on your data range
        self.ax1.set_ylim(config.data["minDB"], config.data["maxDB"])
        self.ax1.set_xlabel("Frequency (GHz)")
        self.ax1.set_ylabel("Magnitude (dB)")
        self.ax1.set_title(
            "Real Time Plot of Frequencies and Magnitudes, "
            + str(config.data["points"])
            + " Points"
        )
        self.ax1.grid(True)
        # Plot 2: Mag Buffer
        self.ax2.set_xlim(1, config.data["bufferSize"])
        self.ax2.set_xlabel("timestep")
        self.ax2.set_ylabel("Resonant Freqnecy point magnitude (dB)")
        self.ax2.set_title("Change in Resonant Freqnency point Magnitude over time")
        self.ax2.grid(True)

        # Plot 3: Frequency Buffer
        self.ax3.set_xlim(1, config.data["bufferSize"])
        self.ax3.set_xlabel("timestep")
        self.ax3.set_ylabel("Resonant Freqnancy (Ghz)")
        self.ax3.set_title("Change in resonant frequency over time")
        self.ax3.grid(True)
        # Plot 4: Temperature (Real)
        self.ax5.set_xlim(1, config.data["bufferSize"])
        self.ax5.set_xlabel("timestep")
        self.ax5.set_ylabel("Temperature °C")
        self.ax5.set_title("Temperature of Sensor (Real)")
        self.ax5.grid(True)

        # Plot 6: Temperature vs Resonance Frequency
        self.ax6.set_xlabel("Temperature (°C)")
        self.ax6.set_ylabel("Resonance Frequency (Hz)")
        self.ax6.legend()
        self.ax6.grid(True)
        # Plot 7: Temperature vs Resonance Frequency
        self.ax7.set_xlabel("Temperature (°C)")
        self.ax7.set_ylabel("Resonance Magnitude (Db)")
        self.ax7.legend()
        self.ax7.grid(True)
        # Plot 8: Resonant Phase over time
        self.ax8.set_xlim(1, config.data["bufferSize"])
        self.ax8.set_xlabel("timestep")
        self.ax8.set_ylabel("Resonant Phase (Deg)")
        self.ax8.legend()
        self.ax8.grid(True)
        # Plot 9: Phase Angle of Signal
        self.ax9.set_xlim(config.data["freqStart"] * 1e9, config.data["freqEnd"] * 1e9)
        self.ax9.set_ylim(-180, 180)
        self.ax9.set_xlabel("Frequency (GHz)")
        self.ax9.set_ylabel("Resonant Phase (Deg)")
        self.ax9.set_title("Phase angle vs Freqnecy")
        self.ax9.legend()
        self.ax9.grid(True)

        # print(self.deltaMaxes)

        plt.grid(True)
        plt.tight_layout()

        return

    def addStat(self, key: str, value, scaler=1, unit="", decimals=3):
        """Add a statistic to the dictionary."""
        filteredValue = value
        filteredKey = key
        if unit != "":
            filteredKey = f"{key} ({unit})"
        if scaler != 1:
            filteredValue = filteredValue * scaler
        if isinstance(value, float):
            print(f"Adding {key} to stats")
            filteredValue = round(filteredValue, decimals)
        self.stats[filteredKey] = filteredValue

    def printStats(self):
        # Create a list of stat strings
        stat_strings = [f"{key}: {value}" for key, value in self.stats.items()]

        # Set the starting position for the text
        x_pos = -0.2  # x position in the middle of the plot
        y_pos = 0.95  # Starting y position near the top of the plot

        # Add each stat string to the axis
        for i, stat in enumerate(stat_strings):
            self.axStats.text(
                x_pos,
                y_pos - i * 0.05,
                stat,
                fontsize=10,
                ha="left",
                transform=self.axStats.transAxes,
            )

    def update(self):
        try:
            tsFile = self.tsList.getLastTouchstone()
        except Exception as e:
            print(e)
            return

        # Remove all previous markers from plots
        for marker in self.markers:
            marker[0].remove()
            marker[1].remove()
        self.markers.clear()

        # Global Stats
        # Shows the resonance frequency during first 10 timesteps (avg)
        self.addStat(
            "Starting Resonance Freqnecy",
            np.array(self.tsList.getResonanceFrequencyList()[0:10]).mean(),
            1e-9,
            "GHz",
        )

        def updateBufferGraph(buffer, ax, line):
            line.set_xdata(range(len(buffer)))  # Plot rolling buffer
            line.set_ydata(buffer)
            ax.set_xlim(1, len(buffer))
            ax.relim()  # Resize the plot to fit new data
            ax.autoscale_view()

        def updateWaterfallGraph(ax, buffer):
            # Water fall update
            ax.clear()
            length = len(buffer[0])
            start_index = max(0, length - self.num_points_time)

            ax.imshow(
                buffer.T,
                cmap="viridis",
                vmin=self.config.data["minDB"],
                vmax=self.config.data["maxDB"],
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                extent=[0, self.config.data["points"], start_index, length],
            )  # Displaying the waterfall plot
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Time")
            ax.set_title("Waterfall Plot")
            ax.set_ylim(max(0, length - self.num_points_time), length)
            numOfTicks = 5
            ax.set_xticks(np.linspace(0, self.config.data["points"], numOfTicks))
            ax.set_xticklabels(
                [
                    f"{label:.1f} GHz"
                    for label in np.linspace(
                        self.config.data["freqStart"],
                        self.config.data["freqEnd"],
                        numOfTicks,
                    )
                ]
            )

        # Line Graphs
        self.line1.set_xdata(tsFile.getFrequencyRange())
        self.line1.set_ydata(tsFile.getDataMagnitude())
        self.line9.set_xdata(tsFile.getFrequencyRange())
        self.line9.set_ydata(tsFile.getPhaseAngle())
        # self.touchstoneBuffer = np.hstack(
        # self.touchstoneBuffer = np.hstack(
        #     (self.touchstoneBuffer, tsFile.getDataMagnitude()[:, np.newaxis])
        # )

        # Super Complex Muli Line graphs
        min_temp, max_temp = self.tsList.getTemperatureRange()
        tempList = np.linspace(min_temp, max_temp, COMPLEX_LINES)
        colors = [plt.cm.autumn(i / (COMPLEX_LINES - 1)) for i in range(COMPLEX_LINES)]
        colors = colors[::-1]  # Reverse the colors for better visibility

        if abs(max_temp - min_temp) > 50:
            print("Data qualifies for multi line plot")
            # Loop through each temperature in the tempList
            for temp, color in zip(tempList, colors):
                # Find the corresponding Touchstone file for the current temperature
                curTsFile = self.tsList.findTouchstoneByTemperature(temp)[0]

                # Check if the Touchstone file exists
                if curTsFile is not None:
                    # Plot frequency vs magnitude
                    self.axMultiLine.plot(
                        curTsFile.getFrequencyRange(),
                        curTsFile.getDataMagnitude(),
                        label=f"Temp: {int(temp)}°C",  # Adding a label for the legend
                        color=color,
                    )
                else:
                    print(f"No Touchstone file found for temperature: {temp}°C")
        else:
            print("Data does not qualify for multi line plot")

        # Adding labels and a legend for clarity
        self.axMultiLine.set_xlabel("Frequency (Hz)")
        self.axMultiLine.set_ylabel("Magnitude")
        self.axMultiLine.grid(True)  # Optional: Add grid for better readability
        # Configure the legend
        legend = self.axMultiLine.legend(
            fontsize="small", loc="upper left"
        )  # Change 'small' to desired size

        # Optionally, shrink the legend box (you may need to adjust based on your plot)
        frame = legend.get_frame()
        frame.set_alpha(0.5)  # Set transparency of the legend background

        updateBufferGraph(
            self.tsList.getResonanceMagnitudeList(),
            self.ax2,
            self.line2,
        )
        updateBufferGraph(
            self.tsList.getResonanceFrequencyList(),
            self.ax3,
            self.line3,
        )
        updateBufferGraph(
            self.tsList.getTemperatureDataList(),
            self.ax5,
            self.line5,
        )
        updateBufferGraph(
            self.tsList.getPhaseDataList(),
            self.ax8,
            self.line8,
        )

        updateWaterfallGraph(self.ax4, self.tsList.getWaterFallDataList())

        # Update Temperature vs Resonance Frequency

        temperature = self.tsList.getTemperatureDataList()
        resonanceFrequency = self.tsList.getResonanceFrequencyList()
        resonanceMagnitude = self.tsList.getResonanceMagnitudeList()
        assert len(temperature) == len(resonanceFrequency)
        ndTemperature = np.array(temperature)
        # Step 1: Identify the minimum and maximum temperatures
        min_temp, max_temp = self.tsList.getTemperatureRange()

        # Step 2: Define weights based on distance from min and max temperatures
        def calculate_weights(temp):
            # Calculate lower and upper 5% thresholds
            lower_bound = np.percentile(temp, 5)
            upper_bound = np.percentile(temp, 95)

            # Create weights
            weights = np.ones_like(temperature)  # Start with equal weights
            weights[temperature <= lower_bound] = 2.0  # Weight for lower 5%
            weights[temperature >= upper_bound] = 2.0  # Weight for upper 5%
            return weights

        # Apply weight calculation
        weightedTemperature = calculate_weights(ndTemperature)

        # Step 3: Fit a weighted linear regression model
        X = ndTemperature.reshape(-1, 1)  # Reshape for sklearn
        y = resonanceFrequency
        weights = weightedTemperature

        model = LinearRegression()
        model.fit(X, y, sample_weight=weights)

        # Get the slope and intercept
        slope = model.coef_[0]
        self.addStat("Resonance Freqnecy Sensitivity", slope, 1e-6, "MHz/°C")
        intercept = model.intercept_

        # Step 4: Print the slope
        print(f"Weighted slope of Frequency with respect to Temperature: {slope}")

        self.ax6.scatter(
            temperature,
            resonanceFrequency,
            color="green",
            label="temp vs resonance",
        )
        # add line of best fit to scatter plot
        x = np.linspace(min_temp, max_temp, 100)
        y = model.predict(x.reshape(-1, 1))
        self.ax6.plot(x, y, color="red")

        self.ax7.scatter(
            temperature,
            resonanceMagnitude,
            color="blue",
            label="temp vs resonance",
        )

        def linear_fit(x, m, b):
            x = np.array(x)
            return m * x + b

        # Perform the curve fit
        # if len(temperature) > 1:

        # params, params_covariance = curve_fit(
        #     linear_fit, temperature, resonanceFrequency
        # )
        # line_fit = linear_fit(temperature, params[0], params[1])

        # # Plot the of best fit on ax6
        # # self.line6.set_data(temperature, line_fit)

        # # Calculate RMSE
        # rmse = np.sqrt(mean_squared_error(resonanceFrequency, line_fit)) Testing123

        self.ax6.set_title("Temperature vs Resonance Frequency")
        self.ax7.set_title("Temperature vs Resonance Mangnitude")
        # Append Marker
        minX = tsFile.getResonanceFrequency()[0]
        minY = tsFile.getResonanceFrequency()[1]
        marker = self.ax1.plot(minX, minY, "ro", label="Largest Change in Magnitude")
        text = self.ax1.text(
            minX,
            minY,
            f"{minX.real/10e8:.2f}GHz" + ", " + f"{minY:.2f}",
            fontsize=12,
            ha="right",
        )
        self.markers.append((marker[0], text))

        # Update stats
        self.printStats()

        # print(start_index, frame)

        return (
            self.line1,
            self.line2,
            self.line3,
            self.line5,
            self.line6,
            self.line7,
            self.line8,
            self.line9,
        )

    def data_gen():
        while True:
            yield None  # This is a placeholder since FuncAnimation requires a generator

    def getPlot(self):
        return self.fig

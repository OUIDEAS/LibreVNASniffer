#!/usr/bin/env python3

from librevna import libreVNA
from vnacommandcenter import VNACommandCenter
import numpy as np
import matplotlib.pyplot as plt
from touchstone import Touchstone as touch
from touchstone import TouchstoneList
from matplotlib.animation import FuncAnimation
from thermocouple import Thermocouple


class VNAPlot:
    def __init__(
        self,
        center,
        thermocouple: Thermocouple,
        tsList: TouchstoneList,
        IFBW,
        FREQSTART,
        FREQEND,
        POINTS,
        HISTORYBUFFERSIZE,
        SIGNAL,
        MAXDB,
        MINDB,
    ):
        np.set_printoptions(precision=10)

        # Initialize plot
        self.fig, axs = plt.subplots(3, 2, dpi=50)
        ax1 = axs[0, 0]
        ax2 = axs[1, 0]
        ax3 = axs[2, 0]
        ax4 = axs[0, 1]
        ax5 = axs[1, 1]
        ax6 = axs[2, 1]

        self.fig.set_size_inches(10, 10)
        (line1,) = ax1.plot([], [], "r-", linewidth=1)
        (line2,) = ax2.plot([], [], linewidth=1, color="blue")
        (line3,) = ax3.plot([], [], linewidth=1, color="green")
        (line5,) = ax5.plot([], [], linewidth=1, color="brown")
        # Generate sample data
        num_points_time = 120  # Two minutes of data
        num_points_frequency = 50
        data = (np.random.rand(num_points_time * 10, num_points_frequency) * 50) - 70
        # Print information about the data
        # print("Data shape:", data.shape)
        # print("Data type:", data.dtype)
        # print("Data range:", np.min(data), np.max(data))
        print(type(MINDB))
        self.fig.colorbar(
            ax4.imshow(
                data,
                cmap="viridis",
                vmin=MINDB,
                vmax=MAXDB,
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                extent=[0, POINTS, 0, num_points_time],
            ),
            ax=ax4,
        )

        ax4.set_label("Magnitude (dB)")

        # Plot 1: Mag Buffer
        ax1.set_xlim(FREQSTART * 1e9, FREQEND * 1e9)
        # Adjust limits based on your data range
        ax1.set_ylim(MINDB, MAXDB)
        ax1.set_xlabel("Frequency (GHz)")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_title(
            "Real Time Plot of Frequencies and Magnitudes, " + str(POINTS) + " Points"
        )
        ax1.grid(True)
        # Plot 2: Mag Buffer
        ax2.set_xlim(1, HISTORYBUFFERSIZE)
        ax2.set_xlabel("timestep")
        ax2.set_ylabel("Resonant Freqnecy point magnitude (dB)")
        ax2.set_title("Change in Resonant Freqnency point Magnitude over time")
        ax2.grid(True)

        # Plot 3: Frequency Buffer
        ax3.set_xlim(1, HISTORYBUFFERSIZE)
        ax3.set_xlabel("timestep")
        ax3.set_ylabel("Resonant Freqnancy (Ghz)")
        ax3.set_title("Change in resonant frequency over time")
        ax3.grid(True)
        # Plot 4: Temperature (Real)
        ax5.set_xlim(1, HISTORYBUFFERSIZE)
        ax5.set_xlabel("timestep")
        ax5.set_ylabel("Temperature °C")
        ax5.set_title("Temperature of Sensor (Real)")
        ax5.grid(True)

        # print(self.deltaMaxes)
        markers = []

        def update(frame):
            # Remove all previous markers from plots
            for marker in markers:
                marker[0].remove()
                marker[1].remove()
            markers.clear()

            # Get new data from VNA
            tsFile = touch(
                center.requestFrequencySweep(
                    -10, IFBW, 1, POINTS, FREQSTART, FREQEND, SIGNAL
                )
            )
            tsFile.addTemperatureData(thermocouple.readTempatureCelsius())
            tsList.addTouchstone(tsFile)

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
                start_index = max(0, length - num_points_time)

                ax.imshow(
                    buffer.T,
                    cmap="viridis",
                    vmin=MINDB,
                    vmax=MAXDB,
                    aspect="auto",
                    interpolation="nearest",
                    origin="lower",
                    extent=[0, POINTS, start_index, length],
                )  # Displaying the waterfall plot
                ax.set_xlabel("Frequency")
                ax.set_ylabel("Time")
                ax.set_title("Waterfall Plot")
                ax.set_ylim(max(0, length - num_points_time), length)
                numOfTicks = 5
                ax.set_xticks(np.linspace(0, POINTS, numOfTicks))
                ax.set_xticklabels(
                    [
                        f"{label:.1f} GHz"
                        for label in np.linspace(FREQSTART, FREQEND, numOfTicks)
                    ]
                )

            line1.set_xdata(tsFile.getFrequencyRange())
            line1.set_ydata(tsFile.getDataMagnitude())
            # self.touchstoneBuffer = np.hstack(
            #     (self.touchstoneBuffer, tsFile.getDataMagnitude()[:, np.newaxis])
            # )

            updateBufferGraph(
                tsList.getResonanceFrequencyList(),
                ax2,
                line2,
            )
            updateBufferGraph(
                tsList.getResonanceMagnitudeList(),
                ax3,
                line3,
            )
            updateBufferGraph(
                tsList.getTemperatureDataList(),
                ax5,
                line5,
            )

            updateWaterfallGraph(ax4, tsList.getWaterFallDataList())

            # Append Marker
            minX = tsFile.getResonanceFrequency()[0]
            minY = tsFile.getResonanceFrequency()[1]
            marker = ax1.plot(minX, minY, "ro", label="Largest Change in Magnitude")
            text = ax1.text(
                minX,
                minY,
                str(np.abs(minX) / 10e8) + ", " + str(minY),
                fontsize=12,
                ha="right",
            )
            markers.append((marker[0], text))

            # print(start_index, frame)

            return (
                line1,
                line2,
                line3,
            )

        def data_gen():
            while True:
                yield None  # This is a placeholder since FuncAnimation requires a generator

        # Create the animation
        self.ani = FuncAnimation(self.fig, update, blit=False, interval=100)

        plt.grid(True)
        plt.tight_layout()

        return

    def getPlot(self):
        return self.fig

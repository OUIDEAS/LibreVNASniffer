import numpy as np
from scipy.fft import ifft
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import csv


class Touchstone:
    def __init__(self, rawData) -> None:
        self.data = np.array(rawData)
        magdB = self.getDataMagnitude()
        minIndex = np.argmin(magdB)
        minValue = (magdB).min()
        self.resonanceFrequency = self.data[minIndex, 0]
        self.resonanceMagnitude = minValue
        self.resonanceComplex = self.data[minIndex, 1]

    def getFrequencyRange(self):
        return self.data[:, 0]

    def getDataMagnitude(self):
        return 20 * np.log10(np.abs(self.data[:, 1]))

    def getTemperatureData(self):
        return self.temperatureData

    def addTemperatureData(self, data):
        self.temperatureData = data

    def getCSVData(self):
        data = []
        data.append(float(self.getResonanceFrequency()[0]) / 10**9)
        data.append(self.getResonanceFrequency()[1])
        data.append(self.getTemperatureData())
        for row in self.data:
            data.extend(row)
        return data

    def getResonanceFrequency(self):
        return self.resonanceFrequency, self.resonanceMagnitude, self.resonanceComplex

    def getIFFT(self):
        freq_range = self.getFrequencyRange()
        num_zeros = int(freq_range[0] / (freq_range[1] - freq_range[0]))

        complex_data = self.data[:, 1]
        padded_data = np.pad(complex_data, (num_zeros, 0), mode="constant")
        mirrored_data = padded_data[::-1].conj()
        rearranged_data = np.concatenate((padded_data, mirrored_data))
        time_domain_signal = ifft(rearranged_data)

        # Generate time indices
        sampling_frequency = 10e9  # Example sampling frequency (10 GHz)
        time_indices = np.arange(len(time_domain_signal)) / sampling_frequency

        # Plot the real part of the time-domain signal
        plt.plot(time_indices, np.real(time_domain_signal))
        plt.title("Time Domain Signal (Real Part)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
        return time_domain_signal

    def __repr__(self):
        return f"Touchstone point{self.getResonanceFrequency()!r}"


class TouchstoneList:
    def __init__(self):
        self.touchstones = []

        self.firstTimestamp = None

    def addTouchstone(self, touchstone):
        if not isinstance(touchstone, Touchstone):
            raise TypeError("Only Touchstone objects can be added.")

        timestamp = datetime.now()
        if self.firstTimestamp is None:
            self.firstTimestamp = datetime.now()
            timestamp = timedelta(0)
        else:
            timestamp = datetime.now() - self.firstTimestamp

        self.touchstones.append((touchstone, timestamp))

    def getTouchstones(self):
        return self.touchstones

    def getResonanceMagnitudeList(self):
        results = [tsfile[0].getResonanceFrequency()[0] for tsfile in self.touchstones]
        return results

    def getResonanceFrequencyList(self):
        results = [tsfile[0].getResonanceFrequency()[1] for tsfile in self.touchstones]
        return results

    def getWaterFallDataList(self):
        numOfPoints = len(self.touchstones[0][0].getFrequencyRange())
        buffer = np.zeros((numOfPoints, 0))
        for tsFile in self.touchstones:
            buffer = np.hstack((buffer, tsFile[0].getDataMagnitude()[:, np.newaxis]))
        return buffer

    def getTemperatureDataList(self):
        results = [tsfile[0].getTemperatureData() for tsfile in self.touchstones]
        return results

    def saveTouchstoneListAsCSV(self, filename):
        data = []
        for touchstone in self.touchstones:
            tsData = touchstone[0].getCSVData()
            tsData.insert(0, touchstone[1])
            data.append(tsData)
        # Write the 2D array to a CSV file

        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)

    def __repr__(self):
        return f"================= Total Touchstones {len(self.touchstones)!r}\n".join(
            [
                f"{touchstone} recorded at {timestamp}"
                for touchstone, timestamp in self.touchstones
            ]
        )

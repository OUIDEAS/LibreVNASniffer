import numpy as np
from scipy.fft import ifft
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import csv
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from dataconfig import DataConfig
import os
import sys


class Touchstone:
    def __init__(self, rawData) -> None:
        self.data = np.array(rawData)
        magdB = self.getDataMagnitude()
        minIndex = np.argmin(magdB)
        minValue = (magdB).min()
        self.resonanceFrequency = self.data[minIndex, 0].real
        self.resonanceMagnitude = minValue
        self.resonanceComplex = self.data[minIndex, 1]
        self.resonancePhase = np.degrees(np.angle(self.resonanceComplex))

    def getFrequencyRange(self):
        return [z.real for z in self.data[:, 0]]

    def getDataMagnitude(self):
        return 20 * np.log10(np.abs(self.data[:, 1]))

    def getTemperatureData(self):
        return self.temperatureData

    def getPhaseAngle(self):
        return np.degrees(np.angle(self.data[:, 1]))

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

    @classmethod
    def loadCSVData(self, row):
        timestamp = row[0]
        resonanceFreq = float(row[1])
        ResonanceMag = float(row[2])
        temperature = float(row[3])

        freq_complex_pairs = []
        for i in range(4, len(row), 2):
            freq = complex(row[i])
            complex_value = complex(
                row[i + 1]
            )  # Assuming the complex value is in a string form that can be converted
            freq_complex_pairs.append((freq, complex_value))
        ts = Touchstone(freq_complex_pairs)
        ts.addTemperatureData(temperature)
        return ts

    def getResonanceFrequency(self):
        return (
            self.resonanceFrequency,
            self.resonanceMagnitude,
            self.resonanceComplex,
            self.resonancePhase,
        )

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
        self.invalidData = 0
        self.firstTimestamp = None
        self.name = None
        self.config = None

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

    def getSlopeAndInterceptOfResonantFreq(self):
        temperature = self.getTemperatureDataList()
        ndTemperature = np.array(temperature)

        def calculate_weights(temp):
            # Calculate lower and upper 5% thresholds
            lower_bound = np.percentile(temp, 5)
            upper_bound = np.percentile(temp, 95)

            # Create weights
            weights = np.ones_like(temperature)  # Start with equal weights
            weights[temperature <= lower_bound] = 2.0  # Weight for lower 5%
            weights[temperature >= upper_bound] = 2.0  # Weight for upper 5%
            return weights

        weightedTemperature = calculate_weights(ndTemperature)

        # Step 3: Fit a weighted linear regression model
        X = ndTemperature.reshape(-1, 1)  # Reshape for sklearn
        y = self.getResonanceFrequencyList()
        weights = weightedTemperature

        model = LinearRegression()
        model.fit(X, y, sample_weight=weights)
        # Compute the line of best fit
        # Get the slope and intercept
        slope = model.coef_[0]
        intercept = model.intercept_

        return slope, intercept

    # Returns R^2 value of the line of best fit
    def getR2(self):
        x = np.array(self.getResonanceFrequencyList())
        y = np.array(self.getTemperatureDataList())
        if len(x) != len(y):
            raise ValueError("x and y must have the same length to compute R²")

        # Fit a simple linear regression model: y = mx + b
        m, b = np.polyfit(x, y, 1)  # Linear regression (degree=1)

        # Predicted y values
        y_pred = m * x + b

        # Compute R² score
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        r_squared = 1 - (ss_res / ss_tot)

        return r_squared

    # returns the resonance freqnecy at temperature
    def getRootFrequency(self, temperature=30):
        slope, intercept = self.getSlopeAndInterceptOfResonantFreq()
        estimatedRootFreqnecy = slope * temperature + intercept
        # print(slope, intercept)
        return estimatedRootFreqnecy

    def getLastTouchstone(self) -> Touchstone:
        if len(self.touchstones) <= 0:
            raise Exception("No Touchstones Avaliable to display")
        else:
            return self.touchstones[-1][0]

    def getWaterFallDataList(self):
        numOfPoints = len(self.touchstones[0][0].getFrequencyRange())
        buffer = np.zeros((numOfPoints, 0))
        for tsFile in self.touchstones:
            buffer = np.hstack((buffer, tsFile[0].getDataMagnitude()[:, np.newaxis]))
        return buffer

    def getTemperatureDataList(self):
        results = [tsfile[0].getTemperatureData() for tsfile in self.touchstones]
        return results

    def getTemperatureRange(self):
        temperature = self.getTemperatureDataList()
        return [min(temperature), max(temperature)]

    def getResonanceFrequencyList(self):
        results = [
            abs(tsfile[0].getResonanceFrequency()[0]) for tsfile in self.touchstones
        ]
        return results

    def getNormalizedResonanceFrequencyList(self):
        slope, intercept = self.getSlopeAndInterceptOfResonantFreq()
        results = [
            abs(tsfile[0].getResonanceFrequency()[0]) for tsfile in self.touchstones
        ]

        # Normalize the resonance freqnecy data using slope and intercept
        results = [(x - intercept) / slope for x in results]

        return results

    def getResonanceMagnitudeList(self):
        results = [tsfile[0].getResonanceFrequency()[1] for tsfile in self.touchstones]
        return results

    def getComplexDataList(self):
        results = [tsfile[0].getResonanceFrequency()[2] for tsfile in self.touchstones]
        return results

    def getPhaseDataList(self):
        results = [tsfile[0].getResonanceFrequency()[3] for tsfile in self.touchstones]
        return results

    def getDistanceDataList(self):
        distanceinMM = self.config.data["distance"]
        # fill up linspace with distanceinMM
        results = [distanceinMM for tsfile in self.touchstones]
        return results

    def saveTouchstoneListAsCSV(self, filename):
        print("saving csv")
        data = []
        for touchstone in self.touchstones:
            tsData = touchstone[0].getCSVData()
            tsData.insert(0, touchstone[1])
            data.append(tsData)
        # Write the 2D array to a CSV file

        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)
        print("csv saved")

    # finds the touchstone closest to the given temperature
    def findTouchstoneByTemperature(self, temperature):
        closestTouchstone = None
        closestDistance = float("inf")
        for touchstone in self.touchstones:
            distance = abs(touchstone[0].getTemperatureData() - temperature)
            if distance < closestDistance:
                closestTouchstone = touchstone
                closestDistance = distance
        return closestTouchstone

    # returns if the TouchstoneList is withing certain quality standards, Will not be trained on if not
    def isQualityDataset(self):
        slope, _ = self.getSlopeAndInterceptOfResonantFreq()
        R2 = self.getR2()
        if slope > 0:
            print("Slope is positive on dataset ", self.name)
            return False
        if R2 < 0.9:
            print("Dataset ", self.name, " is low quality with R2 of ", R2)
            return False

        return True

    @staticmethod
    def loadTouchstoneListFromCSV(filename) -> "TouchstoneList":
        fieldnames = ["timestamp", "resonanceFreq", "resonanceMag", "temp"]
        tsl = TouchstoneList()
        with open(filename, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                ts = Touchstone.loadCSVData(row)
                tsl.addTouchstone(ts)
        # Isolate just the csv filename
        tsl.name = filename.split("/")[-1]
        dir = filename.rsplit("/", 1)[0]
        tsl.config = DataConfig.loadConfig(dir)
        print("Loaded touchstone list from", dir + "/" + tsl.name)
        return tsl

    def __repr__(self):
        return f"================= total touchstones {len(self.touchstones)!r}\n".join(
            [
                f"{touchstone} recorded at {timestamp}"
                for touchstone, timestamp in self.touchstones
            ]
        )

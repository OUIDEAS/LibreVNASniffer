from vnacommandcenter import VNACommandCenter
from thermocouple import Thermocouple
from touchstone import Touchstone, TouchstoneList
from dataconfig import DataConfig
import numpy as np

TIMESTEPS_CUTOFF = 50


class DataCenter:
    def __init__(self):
        try:
            self.connectToThermocouple()
            self.connectToVNA()
        except Exception:
            raise
        return

    # Checks if the resonance frequency is within 4 Standard Deviations of the mean for the current tsList
    def checkForInvalidData(self, tsFile: Touchstone, tsList: TouchstoneList):
        if len(tsList.touchstones) < TIMESTEPS_CUTOFF:
            return False

        mean = np.mean(tsList.getResonanceFrequencyList())
        std = np.std(tsList.getResonanceFrequencyList())
        distance = abs(tsFile.getResonanceFrequency()[0] - mean)
        if distance > 6 * std:
            print("\033[91mError: Collected Data too far from Mean!\033[0m")
            print(
                "Invalid Data with resonance freqnecy of ",
                tsFile.getResonanceFrequency()[0] / 1e9,
                " Was removed for being too far from the mean",
                mean / 1e9,
                " by ",
                distance / 1e9,
                "where the max is",
                10 * std / 1e9,
                " away",
            )
            return True
        else:
            return False

    def getData(self, tsList, config: DataConfig):
        # Get new data from VNA
        print("Polling VNA for new data")
        data = self.vna.requestFrequencySweep(
            -2,
            config.data["IFBW"],
            1,
            config.data["points"],
            config.data["freqStart"],
            config.data["freqEnd"],
            config.data["signalName"],
        )
        if data is None:
            return
        tsFile = Touchstone(data)
        print("Polling Thermocouple for new data")
        temp = self.thermocouple.readTempatureCelsius()
        print("Temperature: ", temp)
        tsFile.addTemperatureData(temp)
        print("Polling Finished, Checking for valid data")
        if self.checkForInvalidData(tsFile, tsList):
            tsList.invalidData += 1
            return
        print("Data is valid, adding to tsList")
        tsList.addTouchstone(tsFile)

    def attachThermocouple(self, thermocouple):
        self.thermocouple = thermocouple

    def attachVNA(self, VNA):
        self.vna = VNA

    def connectToVNA(self):
        try:
            self.vna = VNACommandCenter("localhost", 19542)
        except Exception:
            result = "cannot connect to VNA, Is LibreVNAGUI Running?"
            print(result)
            raise

    def connectToThermocouple(self):
        self.thermocouple = Thermocouple()
        try:
            self.thermocouple.connect()
        except Exception:
            result = "cannot connect to thermocouple"
            print(result)
            raise
        return

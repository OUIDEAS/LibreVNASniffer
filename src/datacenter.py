from vnacommandcenter import VNACommandCenter
from thermocouple import Thermocouple
from touchstone import Touchstone, TouchstoneList
from dataconfig import DataConfig


class DataCenter:
    def __init__(self):
        try:
            self.connectToThermocouple()
            self.connectToVNA()
        except Exception:
            raise
        return

    def getData(self, tsList, config: DataConfig):
        # Get new data from VNA
        tsFile = Touchstone(
            self.vna.requestFrequencySweep(
                -10,
                config.IFBW,
                1,
                config.points,
                config.freqStart,
                config.freqEnd,
                config.signalName,
            )
        )
        tsFile.addTemperatureData(self.thermocouple.readTempatureCelsius())
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

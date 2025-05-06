import json
from datetime import datetime
import os


class DataConfig:
    def __init__(
        self=None,
        freqStart=None,
        freqEnd=None,
        points=None,
        signalName=None,
        maxDB=None,
        minDB=None,
        IFBW=None,
        bufferSize=None,
        distance=None,
    ):
        self.data = {
            "freqStart": freqStart,
            "freqEnd": freqEnd,
            "points": points,
            "signalName": signalName,
            "maxDB": maxDB,
            "minDB": minDB,
            "IFBW": IFBW,
            "bufferSize": bufferSize,
            # Distance from sensor in mm
            "distance": distance,
        }

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid configuration parameter")

    def getNumberOfParameters(self):
        return len(self.data)

    def display_config(self):
        for key, value in self.data.items():
            print(f"{key}: {value}")

    def saveConfig(self, dir):
        print("Saving config")
        filename = dir + "/config.json"
        with open(filename, "w") as file:
            json.dump(self.data, file, indent=4)
        pass
        print("config saved")

    def loadConfig(filename):
        # Load the object from a JSON file
        # If the file is not found, return None
        filename = filename + "/config.json"
        config = DataConfig()
        try:
            with open(filename, "r") as file:
                data = json.load(file)
                config.data = data
                return config
        except FileNotFoundError:
            return None

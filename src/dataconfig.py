class DataConfig:
    def __init__(
        self, freqStart, freqEnd, points, signalName, maxDB, minDB, IFBW, bufferSize
    ):
        self.freqStart = freqStart
        self.freqEnd = freqEnd
        self.points = points
        self.signalName = signalName
        self.maxDB = maxDB
        self.minDB = minDB
        self.IFBW = IFBW
        self.bufferSize = bufferSize

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid configuration parameter")

    def display_config(self):
        config = {
            "sampling_rate": self.sampling_rate,
            "duration": self.duration,
            "sensor_type": self.sensor_type,
            "threshold": self.threshold,
            "output_dir": self.output_dir,
        }
        for key, value in config.items():
            print(f"{key}: {value}")

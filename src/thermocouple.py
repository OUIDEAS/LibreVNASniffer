import uldaq
from uldaq import (
    get_daq_device_inventory,
    DaqDevice,
    InterfaceType,
    TcType,
    TempScale,
)
import time


class Thermocouple:
    def __init__(self, device_index=0, tc_type=TcType.K):
        self.device_index = device_index
        self.tc_type = tc_type
        self.daq_device = None

    def connect(self):
        # Get a list of available DAQ devices
        devices = get_daq_device_inventory(InterfaceType.USB)
        if not devices:
            raise Exception("No DAQ devices found")

        # Create a DAQ device object
        self.daq_device = DaqDevice(devices[self.device_index])

        # Establish a connection to the DAQ device
        self.daq_device.connect()
        self.daq_device.flash_led(3)

        self.ai_device = self.daq_device.get_ai_device()
        self.ai_config = self.ai_device.get_config()
        self.ai_config.set_chan_tc_type(0, TcType.T)

    def readTempatureCelsius(self):
        channel = 0  # Assuming the thermocouple is connected to the first channel
        temp = TempScale.CELSIUS
        if not self.daq_device:
            raise Exception("Device not connected")

        # Read the temperature data
        temperature = self.ai_device.t_in(channel, temp)
        self.daq_device.flash_led(1)
        return temperature

    def readTempatureFahrenheit(self):
        return (self.readTempatureCelsius() * 9 / 5) + 32

    def disconnect(self):
        if self.daq_device:
            self.daq_device.disconnect()
            self.daq_device.release()


if __name__ == "__main__":
    # Example usage:
    thermocouple = Thermocouple()

    try:
        thermocouple.connect()

        while True:
            temperature = thermocouple.readTempatureCelsius()
            print(f"Current Temperature: {temperature:.2f} °C")
            time.sleep(1)  # Delay for 1 second before the next reading

    except KeyboardInterrupt:
        print("Terminated by user")

    finally:
        thermocouple.disconnect()

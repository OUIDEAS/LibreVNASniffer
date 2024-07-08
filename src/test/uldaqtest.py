import uldaq
from uldaq import (
    get_daq_device_inventory,
    DaqDevice,
    InterfaceType,
    AiDevice,
    AiInputMode,
    Range,
    AInFlag,
    TcType,
    TempScale,
    TInFlag,
)
import time


def celsius_to_fahrenheit(celsius):
    fahrenheit = (celsius * 9 / 5) + 32
    return fahrenheit


def main():
    # Get a list of available DAQ devices
    devices = get_daq_device_inventory(InterfaceType.USB)
    if not devices:
        raise Exception("No DAQ devices found")

    # Create a DAQ device object
    daq_device = DaqDevice(devices[0])
    ai_device = daq_device.get_ai_device()

    # Establish a connection to the DAQ device
    daq_device.connect()

    # Define the parameters for temperature reading
    channel = 0  # Assuming the thermocouple is connected to the first channel
    temp = TempScale.CELSIUS
    ai_config = ai_device.get_config()
    ai_config.set_chan_tc_type(0, TcType.T)

    try:
        while True:
            # Read the temperature data
            data = ai_device.t_in(channel, temp)
            print(f"Temperature: {data:.2f} °C {celsius_to_fahrenheit(data):.2f} F")
            # time.sleep(10)  # Delay for 1 second before the next reading

    except KeyboardInterrupt:
        print("Terminated by user")

    finally:
        # Disconnect the DAQ device
        daq_device.disconnect()
        daq_device.release()


if __name__ == "__main__":
    main()

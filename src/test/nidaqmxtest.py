import nidaqmx

devices = [dev.name for dev in nidaqmx.system.System.local().devices]
print("Devices")
print(devices)
if not devices:
    exit()
task = nidaqmx.Task()
task.ai_channels.add_ai_thrmcpl_chan("TC01/ai0")

task.start()
value = task.read()
print(round(value, 1))
task.stop()
task.close()

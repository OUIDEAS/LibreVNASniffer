# Class dedicated to compossing different VNA commands into premade functions, like frequency sweeps
from librevna import libreVNA
import time


class VNACommandCenter:
    def __init__(self, ip, port):
        try:
            self.vna = libreVNA(ip, port)
        except Exception as e:
            print("Failed to connect")
            raise e
        # Make sure we are connecting to a device (just to be sure, with default settings the LibreVNA-GUI auto-connects)
        self.vna.cmd(":DEV:CONN")
        return

    def requestFrequencySweep(
        self, powerLevel, IFBW, average, numOfPoints, startFreq, stopFreq, signal
    ):
        # switch to VNA mode, setup the sweep parameters
        # print("Setting up the sweep...")
        self.vna.cmd(":DEV:MODE VNA")
        self.vna.cmd(":VNA:SWEEP FREQUENCY")
        self.vna.cmd(":VNA:STIM:LVL " + str(powerLevel))
        self.vna.cmd(":VNA:ACQ:IFBW " + str(IFBW))
        self.vna.cmd(":VNA:ACQ:AVG " + str(average))
        self.vna.cmd(":VNA:ACQ:POINTS " + str(numOfPoints))
        self.vna.cmd(":VNA:FREQuency:START " + (str(int(startFreq * 10**9))))
        self.vna.cmd(":VNA:FREQuency:STOP " + (str(int(stopFreq * 10**9))))

        # wait for the sweep to finish
        # print("Waiting for the sweep to finish...")
        while self.vna.query(":VNA:ACQ:FIN?") == "FALSE":
            time.sleep(0.1)

        # grab the data of trace S11
        # print("Reading trace data...")
        data = self.vna.query(":VNA:TRACE:DATA? " + signal)
        return self.vna.parse_VNA_trace_data(data)

    def isConnected(self):
        dev = self.vna.query(":DEV:CONN?")
        if dev == "Not connected":
            print("Not connected to any device, aborting")
            return False
        else:
            print("Connected to " + dev)
            return True

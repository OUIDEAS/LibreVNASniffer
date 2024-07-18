import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from vnacommandcenter import VNACommandCenter
from vnaplot import VNAPlot
from touchstone import TouchstoneList
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pickle
from thermocouple import Thermocouple
from dataconfig import DataConfig
from datacenter import DataCenter
from matplotlib.animation import FuncAnimation


# Base Gui Application
class Application:
    def __init__(self):
        self.tsList = TouchstoneList()
        self.PADDING = 2
        self.root = tk.Tk()
        self.root.title("VNA Frequency Analysis")
        self.root.geometry("1920x1080")

        self.buttonFrame = tk.Frame()
        self.buttonFrame.grid(row=0, column=0, padx=10, pady=10)
        self.graphFrame = tk.Frame()
        self.graphFrame.grid(row=0, column=1, padx=10, pady=10)

        self.isPaused = False
        self.fig, _ = plt.subplots()

        defaultConfig = DataConfig(
            freqStart=3.0,
            freqEnd=4.0,
            points=100,
            signalName="S11TG",
            maxDB=0,
            minDB=-70,
            IFBW=1000,
            bufferSize=10,
        )

        # Create and place input labels and entry boxes
        self.create_input_box("Frequency start:", 0, defaultConfig.freqStart)
        self.create_input_box("Frequency end:", 1, defaultConfig.freqEnd)
        self.create_input_box("Number of points on a graph:", 2, defaultConfig.points)
        self.create_input_box("Buffer size:", 3, defaultConfig.bufferSize)
        self.create_input_box("Signal name (S**):", 4, defaultConfig.signalName)
        self.create_input_box("Max db:", 5, defaultConfig.maxDB)
        self.create_input_box("Min db:", 6, defaultConfig.minDB)
        self.create_input_box("IFBW:", 7, defaultConfig.IFBW)

        # Add an Analyze button
        self.analyze_button = tk.Button(
            self.buttonFrame, text="Analyze", command=self.analyze_frequency
        )
        self.printTouchstoneListButton = tk.Button(
            self.buttonFrame, text="PrintTsList", command=self.printTouchstones
        )
        self.saveRunButton = tk.Button(
            self.buttonFrame, text="Save Run", command=self.saveRun
        )
        self.pauseRunButton = tk.Button(
            self.buttonFrame, text="Pause Run", command=self.toggleRun
        )
        self.openFileExplorerButton = tk.Button(
            self.buttonFrame, text="Open saved CSV", command=self.openFileExplorer
        )

        self.pauseRunButton.grid(row=8, column=0, columnspan=1, pady=self.PADDING)
        self.analyze_button.grid(row=8, column=1, columnspan=1, pady=self.PADDING)
        self.printTouchstoneListButton.grid(
            row=9, column=0, columnspan=1, pady=self.PADDING
        )
        self.saveRunButton.grid(row=9, column=1, columnspan=1, pady=self.PADDING)
        self.openFileExplorerButton.grid(row=10, column=0, pady=self.PADDING)
        # Add a text widget to display results
        self.result_text = tk.Text(self.buttonFrame, height=10, width=50)
        self.result_text.grid(row=11, column=0, columnspan=2, pady=self.PADDING)

    def create_input_box(self, label_text, row, defaultValue):
        label = tk.Label(self.buttonFrame, text=label_text)
        label.grid(row=row, column=0, padx=10, pady=self.PADDING, sticky="e")
        entry = tk.Entry(self.buttonFrame)
        entry.grid(row=row, column=1, padx=10, pady=self.PADDING, sticky="w")
        entry.insert(0, defaultValue)
        setattr(self, f"entry_{row}", entry)

    def printTouchstones(self):
        print(self.tsList)

    def toggleRun(self):
        if self.isPaused:
            self.resumeRun()
        else:
            self.pauseRun()

    def resumeRun(self):
        self.isPaused = False
        self.pauseRunButton.config(text="Pause Run")
        if hasattr(self, "ani"):
            self.ani.resume()

    def pauseRun(self):
        self.isPaused = True
        self.pauseRunButton.config(text="Resume Run")
        if hasattr(self, "ani"):
            self.ani.pause()

    def saveRun(self):
        print("File Saving....")
        print(os.getcwd())
        now = datetime.now()
        baseDirectory = "./data"
        prefix = "figure_"
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        # Create the directory structure
        directory = f"{baseDirectory}/run-{timestamp}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Create the filename
        file_name = f"{prefix}{timestamp}.png"
        pickleFileName = f"pickle_{timestamp}.pkl"
        csvFileName = f"csv_{timestamp}.csv"

        # Full path to the file
        file_path = f"{directory}/{file_name}"
        pickleFilePath = f"{directory}/{pickleFileName}"
        csvFilePath = f"{directory}/{csvFileName}"

        # Save the figure
        self.fig.savefig(file_path)
        with open(pickleFilePath, "wb") as file:
            pickle.dump(self.tsList, file)
        print("File Saved")

        # Save csv Data
        self.tsList.saveTouchstoneListAsCSV(csvFilePath)

        return file_path

    def openFileExplorer(self):
        file_path = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if file_path:
            try:
                print(file_path)
                measurements = TouchstoneList.loadTouchstoneListFromCSV(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read CSV file: {e}")
        self.tsList = measurements
        config = self.getConfigFromUser()
        config.freqStart = self.tsList.getLastTouchstone().getFrequencyRange()[0] / 10e8
        config.freqEnd = self.tsList.getLastTouchstone().getFrequencyRange()[-1] / 10e8
        config.points = len(self.tsList.getLastTouchstone().getFrequencyRange())
        self.displayPlot(config)
        return

    def displayPlot(self, config):
        self.VNAplot = VNAPlot(self.tsList, config)
        print("===================================================================")
        self.fig = self.VNAplot.getPlot()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graphFrame)
        self.canvas.draw()

        self.canvas.get_tk_widget().grid(row=0, column=3)
        # creating the Matplotlib toolbar
        toolbarFrame = tk.Frame(self.graphFrame)
        toolbarFrame.grid(row=1, column=3)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbarFrame)
        toolbar.update()

        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().grid(row=2, column=3)
        self.VNAplot.update()

    def mainloop(self, frame):
        self.dataCenter.getData(self.tsList, self.getConfigFromUser())
        self.VNAplot.update()
        return

    def startLoop(self):
        self.ani = FuncAnimation(self.fig, self.mainloop, blit=False, repeat=False)
        self.resumeRun()

    def getConfigFromUser(self) -> DataConfig:
        # Retrieve input values
        frequency_start = float(self.entry_0.get())
        frequency_end = float(self.entry_1.get())
        num_points = int(self.entry_2.get())
        buffer_size = int(self.entry_3.get())
        signal_name = self.entry_4.get()
        max_db = float(self.entry_5.get())
        min_db = float(self.entry_6.get())
        IFBW = int(self.entry_7.get())
        result = (
            f"Frequency start: {frequency_start}\n"
            f"Frequency end: {frequency_end}\n"
            f"Number of points on a graph: {num_points}\n"
            f"Buffer size: {buffer_size}\n"
            f"Signal name (S**): {signal_name}\n"
            f"Max db: {max_db}\n"
            f"Min db: {min_db}\n"
        )
        config = DataConfig(
            frequency_start,
            frequency_end,
            num_points,
            signal_name,
            max_db,
            min_db,
            IFBW,
            buffer_size,
        )
        return config

    def analyze_frequency(self):
        try:
            self.dataCenter = DataCenter()
        except Exception:
            self.result_text.insert(
                tk.END, "Cannot connect to devices, are they plugged in?" + "\n"
            )
            return

        self.displayPlot(self.getConfigFromUser())
        self.startLoop()


if __name__ == "__main__":
    app = Application()
    app.root.mainloop()

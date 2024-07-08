import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from vnacommandcenter import VNACommandCenter
from vnaplot import VNAPlot
from touchstone import TouchstoneList
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pickle
from thermocouple import Thermocouple


class Application:
    def __init__(self):
        self.tsList = TouchstoneList()
        self.PADDING = 2
        self.root = tk.Tk()
        self.root.title("VNA Frequency Analysis")
        self.root.geometry("1920x1080")
        self.isPaused = False
        self.fig, _ = plt.subplots()

        # Create and place input labels and entry boxes
        self.create_input_box("Frequency start:", 0, 3.0)
        self.create_input_box("Frequency end:", 1, 4.0)
        self.create_input_box("Number of points on a graph:", 2, 100)
        self.create_input_box("Buffer size:", 3, 10)
        self.create_input_box("Signal name (S**):", 4, "S11TG")
        self.create_input_box("Max db:", 5, 0)
        self.create_input_box("Min db:", 6, -70)
        self.create_input_box("IFBW:", 7, 1000)

        # Add an Analyze button
        self.analyze_button = tk.Button(
            self.root, text="Analyze", command=self.analyze_frequency
        )
        self.printTouchstoneListButton = tk.Button(
            self.root, text="PrintTsList", command=self.printTouchstones
        )
        self.saveRunButton = tk.Button(self.root, text="Save Run", command=self.saveRun)
        self.pauseRunButton = tk.Button(
            self.root, text="Pause Run", command=self.toggleRun
        )

        self.pauseRunButton.grid(row=8, column=0, columnspan=1, pady=self.PADDING)
        self.analyze_button.grid(row=8, column=1, columnspan=1, pady=self.PADDING)
        self.printTouchstoneListButton.grid(
            row=9, column=0, columnspan=1, pady=self.PADDING
        )
        self.saveRunButton.grid(row=9, column=1, columnspan=1, pady=self.PADDING)

        # Add a text widget to display results
        self.result_text = tk.Text(self.root, height=10, width=50)
        self.result_text.grid(row=10, column=0, columnspan=2, pady=self.PADDING)

    def create_input_box(self, label_text, row, defaultValue):
        label = tk.Label(self.root, text=label_text)
        label.grid(row=row, column=0, padx=10, pady=self.PADDING, sticky="e")
        entry = tk.Entry(self.root)
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
        if hasattr(self, "VNAPlot"):
            self.VNAplot.ani.resume()

    def pauseRun(self):
        self.isPaused = True
        self.pauseRunButton.config(text="Resume Run")
        if hasattr(self, "VNAPlot"):
            self.VNAplot.ani.pause()

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

    def analyze_frequency(self):
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
        try:
            self.vna = VNACommandCenter("localhost", 19542)
        except Exception:
            result = "cannot connect to VNA, Is LibreVNAGUI Running?"
            self.result_text.insert(tk.END, result + "\n")
            return

        self.thermocouple = Thermocouple()
        try:
            self.thermocouple.connect()
        except Exception:
            result = "cannot connect to thermocouple"
            self.result_text.insert(tk.END, result + "\n")
            return

        self.VNAplot = VNAPlot(
            self.vna,
            self.thermocouple,
            self.tsList,
            IFBW,
            frequency_start,
            frequency_end,
            num_points,
            buffer_size,
            signal_name,
            max_db,
            min_db,
        )
        print("===================================================================")
        self.fig = self.VNAplot.getPlot()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()

        self.canvas.get_tk_widget().grid(row=0, column=3)
        # creating the Matplotlib toolbar
        toolbarFrame = tk.Frame(self.root)
        toolbarFrame.grid(row=1, column=3)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbarFrame)
        toolbar.update()

        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().grid(row=2, column=3)
        # self.fig.show()


if __name__ == "__main__":
    app = Application()
    app.root.mainloop()

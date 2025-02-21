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
import time
import glob
from markdownEditor import MarkdownEditor


# Base Gui Application
class Application:
    def __init__(self):
        self.start_time = time.time()  # Start timer

        self.tsList = TouchstoneList()
        self.PADDING = 2

        self.removedPoints = 0

        self.root = tk.Tk()
        self.root.title("VNA Frequency Analysis")
        self.root.geometry("1920x1080")
        # Set up the window close protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Two main frames, One for control panel and the other for the Graphs
        self.toolsFrame = tk.Frame(self.root)
        self.toolsFrame.grid(row=0, column=0, padx=10, pady=10)
        self.graphFrame = tk.Frame(self.root)
        self.graphFrame.grid(row=0, column=1, padx=10, pady=10)

        # Create frames for buttons, inputs, and results
        self.buttonFrame = tk.Frame(self.toolsFrame)
        self.buttonFrame.grid(row=0, column=0, padx=10, pady=10)
        self.markdownEditor = MarkdownEditor(self.toolsFrame)
        self.markdownEditor.grid(row=1, column=0, pady=self.PADDING)
        self.textInputs = tk.Frame(self.toolsFrame)
        self.textInputs.grid(row=2, column=0, pady=self.PADDING)
        self.resultsFrame = tk.Frame(self.toolsFrame)
        self.resultsFrame.grid(row=3, column=0, pady=self.PADDING)

        self.isPaused = False
        self.fig, _ = plt.subplots()

        # Keeps track of the project directory
        self.projectDirectory = None

        defaultConfig = DataConfig(
            freqStart=3.0,
            freqEnd=4.0,
            points=4000,
            signalName="S21TG",
            maxDB=0,
            minDB=-70,
            IFBW=1000,
            bufferSize=10,
            distance=None,
        )
        # Create and place input labels and entry boxes
        inputBoxCount = 0
        for key, value in defaultConfig.data.items():
            if value != None:
                self.create_input_box(
                    self.textInputs, key, inputBoxCount, defaultConfig.data[key]
                )
                inputBoxCount += 1
            else:
                self.create_input_box(self.textInputs, key, inputBoxCount)
                inputBoxCount += 1

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
        buttons = [
            self.analyze_button,
            self.printTouchstoneListButton,
            self.saveRunButton,
            self.pauseRunButton,
            self.openFileExplorerButton,
        ]
        buttonIndex = 0
        for button in buttons:
            button.grid(row=(buttonIndex // 2), column=buttonIndex % 2)
            buttonIndex += 1

        # Add a text widget to display results
        self.result_text = tk.Text(self.resultsFrame, height=10, width=50)
        self.result_text.grid(
            row=0,
            column=0,
            columnspan=2,
            pady=self.PADDING,
        )
        # Markdown Editor (Below Button Frame)

    def create_input_box(self, root, label_text, row, defaultValue=None):
        label = tk.Label(root, text=label_text)
        label.grid(row=row, column=0, padx=10, pady=self.PADDING, sticky="e")
        entry = tk.Entry(root)
        entry.grid(row=row, column=1, padx=10, pady=self.PADDING, sticky="w")
        if defaultValue != None:
            entry.insert(0, defaultValue)
        setattr(self, f"entry_{label_text}", entry)

    def printTouchstones(self):
        print(self.tsList)

    def toggleRun(self):
        if self.isPaused:
            self.resumeRun()
        else:
            self.pauseRun()

    def on_close(self):
        print("Window is closing. Terminating the program.")
        self.root.quit()  # This will stop the Tkinter event loop
        self.root.destroy()  # Close the Tkinter window

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

    def saveExistingProject(self):
        figureDir = glob.glob(os.path.join(self.projectDirectory, "figure_*.png"))
        notesDir = glob.glob(os.path.join(self.projectDirectory, "notes.md"))
        self.saveFigure(figureDir[0])
        self.saveMarkdown(notesDir[0])
        print("===File Updated===")
        return self.projectDirectory

    def saveRun(self):
        if self.projectDirectory:
            messagebox.showinfo("Heads Up", "A project is already open, saving to that")
            return self.saveExistingProject()

        print("File Saving....")
        print(os.getcwd())
        now = datetime.now()
        prefix = "figure_"
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        directory = self.getRunDirectory()
        # Create the filename
        figureFileName = f"{prefix}{timestamp}.png"
        pickleFileName = f"pickle_{timestamp}.pkl"
        csvFileName = f"csv_{timestamp}.csv"

        # Full path to the file
        figureFilePath = f"{directory}/{figureFileName}"
        pickleFilePath = f"{directory}/{pickleFileName}"
        csvFilePath = f"{directory}/{csvFileName}"

        self.saveFigure(figureFilePath)
        self.savePickle(pickleFilePath)
        self.tsList.saveTouchstoneListAsCSV(csvFilePath)
        self.getConfigFromUser().saveConfig(directory)
        print("===File Saved===")
        # show file saved info prompt
        messagebox.showinfo("File Saved", f"File saved to {directory}")
        return directory

    def saveFigure(self, dir):
        # Save the figure
        print("Saving figure")
        self.fig.savefig(dir)
        print("figure saved")

    def savePickle(self, dir):
        # Save the figure
        print("Saving pickle")
        with open(dir, "wb") as file:
            pickle.dump(self.tsList, file)
        print("pickle saved")

    def saveMarkdown(self, dir):
        # Save the Markdown file
        print("Saving Markdown")
        self.markdownEditor.save_file(dir)
        print("Markdown saved")

    def getRunDirectory(self):
        now = datetime.now()
        baseDirectory = "./data"
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        # Create the directory structure
        directory = f"{baseDirectory}/run-{timestamp}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def openFileExplorer(self):
        # if self.projectDirectory:
        #     messagebox.showerror("Error", "A project is already open")
        #     return

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
                print("\033[91mError: Cant Read CSV File!\033[0m")
        self.tsList = measurements
        modified_dir_string = file_path.rsplit("/", 1)[0]
        print(modified_dir_string)
        self.projectDirectory = modified_dir_string
        # Load Markdown Data

        self.markdownEditor.load_file(self.projectDirectory)
        # Load Config data
        config = DataConfig.loadConfig(self.projectDirectory)
        config.data["freqStart"] = (
            self.tsList.getLastTouchstone().getFrequencyRange()[0] / 10e8
        )
        config.data["freqEnd"] = (
            self.tsList.getLastTouchstone().getFrequencyRange()[-1] / 10e8
        )
        config.data["points"] = len(self.tsList.getLastTouchstone().getFrequencyRange())

        self.displayPlot(config)
        # Load existing Figures
        files = glob.glob(os.path.join(self.projectDirectory, "figure_*.png"))
        if files:
            print(f"Found figure, overwritting {files[0]}")
            self.saveFigure(files[0])

        return

    def displayPlot(self, config):
        self.VNAplot = VNAPlot(self.tsList, config)
        print("===================================================================")
        self.fig = self.VNAplot.getPlot()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graphFrame)
        self.canvas.draw()

        self.canvas.get_tk_widget().grid(row=0, column=0)
        # creating the Matplotlib toolbar
        toolbarFrame = tk.Frame(self.graphFrame)
        toolbarFrame.grid(row=1, column=0)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbarFrame)
        toolbar.update()

        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().grid(row=2, column=0)
        self.VNAplot.update()

    def mainloop(self, frame):
        elapsed_time = time.time() - self.start_time  # End timer
        print(f"Figure Render time: {elapsed_time}")
        print("(gui.py) Starting Loop")
        print("(gui.py) Getting Config")

        config = self.getConfigFromUser()
        print("(gui.py) Getting Data")

        # Print time to poll data
        dataPollStart = time.time()
        self.dataCenter.getData(self.tsList, config)
        dataPollEnd = time.time()
        print(f"Data Poll Time: {dataPollEnd - dataPollStart}")
        print("(gui.py) updating plot Data")
        self.VNAplot.update()
        elapsed_time = time.time() - self.start_time  # End timer
        print("(gui.py) Loop Complete, took ", elapsed_time, "seconds")
        self.start_time = time.time()  # Start timer
        return

    def startLoop(self):
        self.ani = FuncAnimation(self.fig, self.mainloop, blit=False, repeat=False)
        self.resumeRun()

    def getConfigFromUser(self) -> DataConfig:
        # Retrieve input values
        frequency_start = float(self.entry_freqStart.get())
        frequency_end = float(self.entry_freqEnd.get())
        signal_name = self.entry_signalName.get()
        max_db = int(self.entry_maxDB.get())
        min_db = int(self.entry_minDB.get())
        IFBW = int(self.entry_IFBW.get())
        buffer_size = int(self.entry_bufferSize.get())
        distance = self.entry_distance.get()
        if distance == "":
            self.result_text.insert(tk.END, "No Default distance" + "\n")
            print("\033[91mError: Input Distance!\033[0m")
            return None
        else:
            print(distance)
            distance = int(distance)
        num_points = int(self.entry_points.get())

        result = (
            f"Frequency start: {frequency_start}\n"
            f"Frequency end: {frequency_end}\n"
            f"Number of points on a graph: {num_points}\n"
            f"Buffer size: {buffer_size}\n"
            f"Signal name (S**): {signal_name}\n"
            f"Max db: {max_db}\n"
            f"Min db: {min_db}\n"
        )
        self.result_text.insert(tk.END, result + "\n")
        config = DataConfig(
            frequency_start,
            frequency_end,
            num_points,
            signal_name,
            max_db,
            min_db,
            IFBW,
            buffer_size,
            distance,
        )
        return config

    def analyze_frequency(self):
        config = self.getConfigFromUser()
        if config == None:
            return
        try:
            self.dataCenter = DataCenter()
        except Exception:
            self.result_text.insert(
                tk.END, "Cannot connect to devices, are they plugged in?" + "\n"
            )
            return
        self.displayPlot(config)
        self.startLoop()


app = Application()
if __name__ == "__main__":
    app.root.mainloop()

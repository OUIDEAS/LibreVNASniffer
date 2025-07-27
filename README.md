

# LibreVNASniffer
Paper: https://www.mdpi.com/1424-8220/25/15/4654

This project is a software package designed to work with the **LibreVNA** and the **Measurement Computing USB-2001-TC** using a **Type T thermocouple**.

There are a few entry points into the project, which are listed below:

---

## `Gui.py`

This is the main window of the program. Running `Gui.py` allows the user to access several functionalities:

### Live Recording of VNA Data

With a computer running **LibreVNAGUI**, and a connected **LibreVNA** and **Measurement Computing USB-2001-TC**, the user will arrive at this UI:

<img width="552" height="1072" alt="image" src="https://github.com/user-attachments/assets/b6281525-5e6b-4d00-b517-c617725ca294" />

#### Buttons

* **Analyze** – Begins recording a `touchstoneList` with the provided parameters. If parameters are missing, an error will occur.
* **Save Run** – Saves the current `touchstoneList` to a CSV in `/data/timestamp/timestamp-csv.csv`.
* **Pause Run** – Halts the collection of touchstones until resumed.
* **Open Saved CSV** – Loads data from a previous recording instead of showing a figure of the currently collected data.
* **Take Screenshot** – Records a single Touchstone file instead of one every 10 seconds. Useful for capturing individual $S_{21}$ readings.

#### Notes

* A note section allows the user to enter run-specific notes.

#### Sweep Configuration

* `freqStart`
* `freqEnd`
* `points`
* `signalName` – Must match a signal name within LibreVNAGUI.
* `maxDB` – Sets the upper y-axis limit of the figures.
* `minDB` – Sets the lower y-axis limit of the figures.
* `IFBW`
* `bufferSize` – *Deprecated; does nothing.*
* `distance` – Distance from the sensor in mm.

#### Console Output

* Error messages will appear here.

---

## `bayesianOpt.py`

*Entry point for Bayesian optimization of model parameters.* 

---

## `candleStickRefactor.py`

Helper Python file responsible for generating summary figures from a collection of datasets. Some of these figures include:

* Resonant Frequency vs Distance
* Accuracy vs Distance
* Initial Frequency vs Distance
* $R^2$ Scores vs Dataset

When this script is run, all data in `/data/` is used, and the figures are saved in `/figures/timestamp/*.png`.

---

## `csvList.py`

Helper file that contains the final 28 datasets in a list called `sensVsDist`.

---

## `hysteresis.py`

Responsible for plotting the hysteresis-over-time figure.

---

## `modeltest.py`

This file runs tests on a selection of models with a set of datasets. It performs:

* Preprocessing
* Model instantiation
* Training
* Performance analysis

Model results are output as figures to `/figures/timestamp/*.png`. These include:

* Model learning curves
* Predicted Temperature vs Real Temperature plots

---

## `screenshotPlotter.py`

Helper file responsible for creating the **time-gating** figures.

---




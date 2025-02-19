# Dataset Class

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from touchstone import Touchstone, TouchstoneList
import tensorflow as tf
from scaler import Scaler
import matplotlib.pyplot as plt

EPOCHS = 750
BUFFER_SIZE = 10000
BATCH_SIZE = 64
TIMESTEPS = 10


class Dataset:
    acceptedFeatures = [
        "normalizedResonanceFrequency",
        # "resonanceFrequency",
        # "deltaResonanceFrequency",
        "resonanceMagnitude",
        "resonancePhase",
        "resonanceReal",
        "resonanceImag",
    ]

    def __init__(
        self, timesteps=TIMESTEPS, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE
    ):
        self.scaler = Scaler()
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.timesteps = timesteps
        # self.epochs = EPOCHS

        self.dataset = None

    # Gets the scalars for the X and y values
    def getScaler(self):
        return self.scaler

    # sets the scalars for the X and y values
    def setScaler(self, scaler):
        self.scaler = scaler

    # Gets features from touchstoneList, Optional filtering of features
    @staticmethod
    def featuresFromTouchstone(touchstone: TouchstoneList):
        feature_map = {
            "resonanceFrequency": touchstone.getResonanceFrequencyList(),
            "normalizedResonanceFrequency": touchstone.getNormalizedResonanceFrequencyList(),
            "resonanceMagnitude": touchstone.getResonanceMagnitudeList(),
            "resonancePhase": touchstone.getPhaseDataList(),
            "resonanceReal": [c.real for c in touchstone.getComplexDataList()],
            "resonanceImag": [c.imag for c in touchstone.getComplexDataList()],
            "deltaResonanceFrequency": np.pad(
                np.diff(touchstone.getResonanceFrequencyList()),
                (1, 0),
                mode="constant",
                constant_values=0,
            ).tolist(),
        }

        # Filter features based on acceptedFeatures
        selected_features = [
            feature_map[key] for key in Dataset.acceptedFeatures if key in feature_map
        ]

        # Convert to NumPy array and transpose
        data = np.array(selected_features).T.tolist()

        X = data
        y = np.array(touchstone.getTemperatureDataList())
        return X, y

    @staticmethod
    def formatFeatures(X, y):
        print("Format Features")
        print(f"X shape: {np.array(X).shape}")
        print(f"y shape: {np.array(y.reshape(-1, 1)).shape}")
        y = y.reshape(-1, 1)
        return X, y

    def concatenateFeatures(self, Xlist, ylist):
        X = np.concatenate(Xlist)
        y = np.concatenate(ylist)
        print("Concatenate Features")
        print(f"X shape: {np.array(X).shape}")
        print(f"y shape: {np.array(y).shape}")
        return X, y

    @staticmethod
    def timestepFeatures(timesteps, X, y):
        X_lstm = []
        y_lstm = []

        for i in range(timesteps, len(X)):
            X_lstm.append(X[i - timesteps : i])
            y_lstm.append(y[i])

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)
        print("TimeStep Features")
        print(f"X shape: {np.array(X_lstm).shape}")
        print(f"y shape: {np.array(y_lstm).shape}")
        return X_lstm, y_lstm

    def flattenFeatures(self, X, y):
        X_reshaped = X.reshape(-1, 5)
        print("Flatten Features")
        print(f"X shape: {np.array(X_reshaped).shape}")
        print(f"y shape: {np.array(y).shape}")
        return X_reshaped, y

    def reshapeFeatures(self, X, y):
        X = X.reshape(-1, self.timesteps, 5)
        print("Reshape Features")
        print(f"X shape: {np.array(X).shape}")
        print(f"y shape: {np.array(y).shape}")
        return X, y

    @staticmethod
    def splitDataset(dataset, validationRatio):
        if dataset is None:
            raise ValueError("Dataset is empty")
        # Create validation dataset
        total_samples = len(dataset)
        validation_size = int(validationRatio * total_samples)
        validation_dataset = dataset.take(validation_size)

        # Create training dataset
        training_dataset = dataset.skip(validation_size)
        return training_dataset, validation_dataset

    def getTimesteplessCopy(self):
        Dataset.print_dataset_info("Before timestep removal", self.dataset)

        # Remove timesteps
        # Map function to reduce timesteps (keep the first timestep)
        def reduce_timesteps(sample, y):
            print(f"Sample shape: {sample.shape}")
            print(f"Index: {y}")
            print(f"Sample: {sample}")
            print(f"Sample 0 shape: {sample[:, 0, :]}")
            return (sample[:, 0, :], y)  # Keep only the first timestep

        # Apply the map function to the dataset
        print(self.dataset)

        datasetCopy = self.dataset.map(reduce_timesteps)

        print(datasetCopy)

        Dataset.print_dataset_info("After timestep removal", datasetCopy)
        return datasetCopy

    # def touchstoneListToDataset(self, touchstone: TouchstoneList):
    #     X, y = self.featuresFromTouchstone(touchstone)
    #     # format features for model
    #     X_lstm, y_lstm = self.formatFeatures(X, y)
    #     dataset = tf.data.Dataset.from_tensor_slices((X_lstm, y_lstm))
    #     return dataset

    @classmethod
    def fromCSVList(
        cls,
        csvList: list,
        timesteps=TIMESTEPS,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
    ):
        plt.figure(figsize=(8, 6))
        # Small dots with transparency

        plt.xlabel("Realative Resonance Frequency")
        plt.ylabel("Temperature (°C)")
        plt.title("Scatter Plot of Resonance Frequency vs Temperature")
        plt.grid(True)
        # Choose a colormap with more distinct colors
        cmap = plt.get_cmap("tab20")  # Or try 'Set3', 'hsv', 'Spectral', etc.
        norm = plt.Normalize(vmin=0, vmax=len(csvList))
        newInstance = cls(timesteps, buffer_size, batch_size)
        formatedFeaturesList = []
        for idx, csv in enumerate(csvList):
            touchstoneList = TouchstoneList.loadTouchstoneListFromCSV(csv)
            slope, _ = touchstoneList.getSlopeAndInterceptOfResonantFreq()
            if slope > 0:
                # Error
                print(f"Error: {csv} has a positive slope")
                exit(1)
            X, y = newInstance.featuresFromTouchstone(touchstoneList)
            # print(f"Size of X: {len(X[:, 0])} Size of y: {len(y)}")
            color = cmap(norm(idx))
            plt.scatter(y, np.array(X)[:, 0], s=5, color=color, alpha=0.7)
            formatedX, formatedY = newInstance.formatFeatures(X, y)
            formatedFeaturesList.append((formatedX, formatedY))
            print("New Features from Touchstone")
            print(f"X shape: {np.array(formatedX).shape}")
            print(f"y shape: {np.array(formatedY).shape}")

        plt.show()
        print(f"Formated Datasets List: {len(formatedFeaturesList)}")
        # print(f"Csv: {csv} has {len(dataset)} samples")
        # print(f"Element spec of {csv}:", dataset.element_spec)
        # datasetList.append(dataset)

        # fit the scaler first
        Xscaling, yScaling = newInstance.concatenateFeatures(
            *zip(*formatedFeaturesList)
        )
        newInstance.scaler.fitAndScaleFeatures(X=None, y=yScaling)
        print("Fitted Scaler")

        # Go through all formated feature tupes and scale,timestep them
        formatedScaleTimestepedFeaturesList = []
        for X, y in formatedFeaturesList:
            _, y = newInstance.scaler.scaleFeatures(X=None, y=y)
            X, _ = newInstance.scaler.fitAndScaleFeatures(X=X, y=None)
            X, y = newInstance.timestepFeatures(timesteps, X, y)
            formatedScaleTimestepedFeaturesList.append((X, y))

        X, y = newInstance.concatenateFeatures(
            *zip(*formatedScaleTimestepedFeaturesList)
        )

        combinedDataset = tf.data.Dataset.from_tensor_slices((X, y))
        # Check the shape of dataset elements
        print("Element spec:", combinedDataset.element_spec)
        # combinedDataset = combinedDataset.batch(1)
        newInstance.print_dataset_info("PreShuffle", combinedDataset)
        combinedDataset = newInstance.shuffleDataset(combinedDataset)
        newInstance.print_dataset_info("PostShuffle", combinedDataset)
        combinedDataset = combinedDataset

        total_samples = len(combinedDataset)
        print(f"Total Batched samples after shuffle: {total_samples}")
        Dataset.print_dataset_info("Combined dataset", combinedDataset)
        newInstance.dataset = combinedDataset
        return newInstance

    @staticmethod
    def r2_score_manual(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        yAvg = np.mean(y_true)
        ss_tot = np.sum((y_true - yAvg) ** 2)

        r2 = 1 - (ss_res / ss_tot)
        return r2

    @staticmethod
    def print_dataset_info(name, dataset):
        # Check if dataset is a TensorFlow Dataset
        print(f"Dataset: {name}")
        if isinstance(dataset, tf.data.Dataset):
            print("\tDataset Type: TensorFlow Dataset")
            # Inspect a single batch
            for x, y in dataset.take(1):  # Take one batch
                print("\tShape of input data:", x.shape)
                print("\tShape of target data:", y.shape)
                print("\tData types of input:", x.dtype)
                print("\tData types of target:", y.dtype)
                # print(
                #     "Sample input data:", x.numpy()
                # )  # Convert tensor to numpy array for display
                # print("Sample target data:", y.numpy())
        # Check if dataset is a NumPy array
        elif isinstance(dataset, np.ndarray):
            print("Dataset Type: NumPy Array")
            print("Shape of dataset:", dataset.shape)
            print("Data type:", dataset.dtype)
            print("First 5 samples:", dataset[:5])  # Print first 5 samples
        # Check if dataset is a pandas DataFrame
        elif isinstance(dataset, pd.DataFrame):
            print("Dataset Type: Pandas DataFrame")
            print("Shape of dataset:", dataset.shape)
            print("Data types of columns:\n", dataset.dtypes)
            print("First 5 rows:\n", dataset.head())  # Print first 5 rows
            print(
                "Summary statistics:\n", dataset.describe()
            )  # Summary stats for numerical columns
        else:
            print("Unsupported dataset type")

    @staticmethod
    def concatenateDatasets(datasets):
        if not datasets:
            raise ValueError("The dataset list is empty.")

        concatenated_dataset = datasets[0]
        # Use functools.reduce to concatenate all datasets in the list
        for i in range(1, len(datasets)):
            concatenated_dataset = concatenated_dataset.concatenate(datasets[i])
        print(
            f"final concatination of {len(datasets)} datasets has {len(concatenated_dataset)} samples"
        )
        return concatenated_dataset

    def shuffleDataset(self, dataset):
        # Shuffle and batch the dataset
        print(dataset)
        # self.plot_tf_dataset(dataset)
        print("Shuffling dataset")
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)
        print("Shuffled")

        # dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Optimize data loading
        # Print dataset elements for verification

        return dataset

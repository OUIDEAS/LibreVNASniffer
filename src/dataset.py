import time
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from touchstone import Touchstone, TouchstoneList
import tensorflow as tf
from scaler import Scaler
import matplotlib.pyplot as plt
import hashlib
import pickle
import math


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
        # "distance",
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
            "distance": touchstone.getDistanceDataList(),
        }

        # confirm that all features have the same length
        for key in feature_map:
            assert len(feature_map[key]) == len(touchstone.getTemperatureDataList())

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
        print(
            "Format Features",
            f"X shape: {np.array(X).shape}",
            f"y shape: {np.array(y.reshape(-1, 1)).shape}",
        )
        y = y.reshape(-1, 1)
        return X, y

    def concatenateFeatures(self, Xlist, ylist):
        X = np.concatenate(Xlist)
        y = np.concatenate(ylist)
        print(
            "Concatenate Features",
            f"X shape: {np.array(X).shape}",
            f"y shape: {np.array(y).shape}",
        )
        return X, y

    @staticmethod
    def timestepFeatures(timesteps, X, y):
        X_lstm = []
        y_lstm = []

        for i in range(timesteps, len(X)):
            timesteped = X[i - timesteps + 1 : i + 1]
            # Reverse order so the current timestep is at index 0
            timesteped = timesteped[::-1]
            # print(f"Timesteped[0]: {timesteped[0]}")
            # print(f"Timesteped: {timesteped}")
            # print(f"Timesteped shape  : {np.array(timesteped).shape}")
            # print(f"X[i]: {X[i]}")
            # print(f"X shape: {np.array(X).shape}")
            assert X[i] == timesteped[0]

            X_lstm.append(timesteped)
            y_lstm.append(y[i])

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)
        print(
            "TimeStep Features",
            f"X shape: {np.array(X_lstm).shape}",
            f"y shape: {np.array(y_lstm).shape}",
        )
        return X_lstm, y_lstm

    def flattenFeatures(self, X, y):
        X_reshaped = X.reshape(-1, 5)
        print(
            "Flatten Features",
            f"X shape: {np.array(X_reshaped).shape}",
            f"y shape: {np.array(y).shape}",
        )
        return X_reshaped, y

    def reshapeFeatures(self, X, y):
        X = X.reshape(-1, self.timesteps, 5)
        print(
            "Reshape Features",
            f"X shape: {np.array(X).shape}",
            f"y shape: {np.array(y).shape}",
        )
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

    @staticmethod
    def getTimesteplessCopy(dataset):
        Dataset.print_dataset_info("Before timestep removal", dataset)

        # Remove timesteps
        # Map function to reduce timesteps (keep the first timestep)
        def reduce_timesteps(sample, y):
            # print(
            #     f"Sample shape: {sample.shape}",
            #     f"Index: {y}",
            #     f"Sample: {sample}",
            #     f"Sample 0 shape: {sample[:, 0, :]}",
            # )
            return (sample[:, 0, :], y)  # Keep only the first timestep

        # Apply the map function to the dataset
        # print(dataset)

        datasetCopy = dataset.map(reduce_timesteps)

        # print(datasetCopy)

        Dataset.print_dataset_info("After timestep removal", datasetCopy)
        return datasetCopy

    # def touchstoneListToDataset(self, touchstone: TouchstoneList):
    #     X, y = self.featuresFromTouchstone(touchstone)
    #     # format features for model
    #     X_lstm, y_lstm = self.formatFeatures(X, y)
    #     dataset = tf.data.Dataset.from_tensor_slices((X_lstm, y_lstm))
    #     return dataset
    @staticmethod
    def compute_hash(csv_list):
        """Compute a hash based on file paths and modification times."""
        hash_obj = hashlib.sha256()
        for file in csv_list:
            if os.path.exists(file):
                mtime = os.path.getmtime(file)  # Last modified time
                hash_obj.update(f"{file}-{mtime}".encode())
            else:
                return None  # If any file is missing, return None to force reload

        # include scaler in hash

        return hash_obj.hexdigest()

    @staticmethod
    def loadCachedDataset(csvList, cacheFile="cachedDatasets.pkl"):
        print("Checking for Cached Dataset")
        hashFile = cacheFile + ".hash"
        currentHash = Dataset.compute_hash(csvList)
        print(f"Current Hash: {currentHash}")
        if currentHash is None:
            return None
        cachePath = "cache/"
        if os.path.exists(cachePath + cacheFile) and os.path.exists(
            cachePath + hashFile
        ):
            with open(cachePath + hashFile, "r") as f:
                cachedHash = f.read().strip()
            if currentHash == cachedHash:
                print("Loading Cached Dataset")
                return pickle.load(open(cachePath + cacheFile, "rb"))
            else:
                print("Hash Mismatch")
        print("No Cached Dataset Found")
        return None

    def saveCachedDataset(self, csvList, cacheFile="cachedDataset"):
        print("Saving Cached Dataset")
        cachePath = "cache/"
        if not os.path.exists(cachePath):
            os.makedirs(cachePath)
        self.dataset.save(cachePath + cacheFile)
        pickle.dump(self.scaler, open(cachePath + cacheFile + "_scaler" + ".pkl", "wb"))
        with open(cachePath + cacheFile + ".hash", "w") as f:
            f.write(self.compute_hash(csvList))
            print("Saved Cached Dataset")
    @staticmethod
    def includeOnlyFeature(X, y, features):
        stringList = Dataset.acceptedFeatures
        indexes = {
            search_string: [i for i, s in enumerate(stringList) if s == search_string]
            for search_string in features
        }
        flat_list = [item for sublist in list(indexes.values()) for item in sublist]

        def removeIndexes(sample):
            return [sample[flat_list[0]]]

        X = list(map(removeIndexes, X))
        print(f"X shape: {np.array(X).shape}")

        return X, y

    @staticmethod
    def includeOnlyFeatures(dataset, features):
        stringList = Dataset.acceptedFeatures
        indexes = {
            search_string: [i for i, s in enumerate(stringList) if s == search_string]
            for search_string in features
        }
        flat_list = [item for sublist in list(indexes.values()) for item in sublist]

        def removeIndexes(sample, y):
            return (
                tf.reshape(sample[:, flat_list[0]], (-1, 1)),
                y,
            )  # Keep only the first timestep

        if flat_list == []:
            ValueError("No features to include")
            return

        datasetCopy = dataset.map(removeIndexes)

        Dataset.print_dataset_info("After feature removal", datasetCopy)
        return datasetCopy

    @classmethod
    def fromCSVList(
        cls,
        csvList: list,
        timesteps=TIMESTEPS,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
    ):
        newInstance = cls(timesteps, buffer_size, batch_size)
        formatedFeaturesList = []
        # cachedInstance = newInstance.loadCachedDataset(csvList)
        # if cachedInstance is not None:
        # return cachedInstance

        plt.figure(figsize=(8, 6))
        from modelplotter import ModelPlotter

        fig, axes = plt.subplots(
            math.ceil(Dataset.numOfFeatures() / 2),
            2,
            figsize=(math.ceil(Dataset.numOfFeatures() / 2) * 8, 16),
        )
        fig.subplots_adjust(hspace=0.3, wspace=0.1)
        # Small dots with transparency

        axes = axes.flatten()

        # Choose a colormap with more distinct colors
        cmap = plt.get_cmap("tab20")  # Or try 'Set3', 'hsv', 'Spectral', etc.
        norm = plt.Normalize(vmin=0, vmax=len(csvList))
        for idx, csv in enumerate(csvList):
            touchstoneList = TouchstoneList.loadTouchstoneListFromCSV(csv)
            if not touchstoneList.isQualityDataset():
                print(
                    f"Skipping {csv} because it is not a quality dataset, has distance of ",
                    touchstoneList.getDistanceDataList()[0],
                )
                continue
            X, y = newInstance.featuresFromTouchstone(touchstoneList)
            # print(f"Size of X: {len(X[:, 0])} Size of y: {len(y)}")
            color = cmap(norm(idx))
            for i, ax in enumerate(axes):
                if i < len(X[0]):
                    ax.set_title(Dataset.acceptedFeatures[i] + " vs Temperature")
                    ax.scatter(y, np.array(X)[:, i], s=5, color=color, alpha=0.7)
                    ax.grid(True)
                    ax.set_xlabel("Temperature (°C)")
                    ax.set_ylabel(Dataset.acceptedFeatures[i])

            formatedX, formatedY = newInstance.formatFeatures(X, y)
            formatedFeaturesList.append((formatedX, formatedY))
            print(
                "New Features from Touchstone",
                f"X shape: {np.array(formatedX).shape}",
                f"y shape: {np.array(formatedY).shape}",
            )

        # plt.show(block=True)
        ModelPlotter.saveFigure(fig, "features_before_scaling")
        print(f"Formated Datasets List: {len(formatedFeaturesList)}")

        fig, axes = plt.subplots(
            math.ceil(Dataset.numOfFeatures() / 2),
            2,
            figsize=(math.ceil(Dataset.numOfFeatures() / 2) * 8, 16),
        )

        fig.subplots_adjust(hspace=0.3, wspace=0.2)  # Adjust spacing between subplots
        axes = axes.flatten()

        formatedScaleTimestepedFeaturesList = []
        for idx, (X, y) in enumerate(formatedFeaturesList):
            X, y = newInstance.scaler.smartScaleFeatures(X, y)
            color = cmap(norm(idx))
            for i, ax in enumerate(axes):
                if i < len(X[0]):
                    ax.set_title("Scaled" + Dataset.acceptedFeatures[i])
                    ax.scatter(y, np.array(X)[:, i], s=5, color=color, alpha=0.7)
                    ax.grid(True)
                    ax.set_xlabel("Temperature (°C)")
                    ax.set_ylabel("scaled" + Dataset.acceptedFeatures[i])
            X, y = newInstance.timestepFeatures(timesteps, X, y)
            formatedScaleTimestepedFeaturesList.append((X, y))

        # plt.show(block=True)
        ModelPlotter.saveFigure(fig, "features_after_scaling")
        print(f"Formated Datasets List: {len(formatedFeaturesList)}")
        X, y = newInstance.concatenateFeatures(
            *zip(*formatedScaleTimestepedFeaturesList)
        )

        combinedDataset = tf.data.Dataset.from_tensor_slices((X, y))
        # Check the shape of dataset elements
        # print("Element spec:", combinedDataset.element_spec)
        # combinedDataset = combinedDataset.batch(1)
        # time.sleep(10)
        newInstance.print_dataset_info("PreShuffle", combinedDataset)
        combinedDataset = newInstance.shuffleDataset(combinedDataset)
        newInstance.print_dataset_info("PostShuffle", combinedDataset)
        combinedDataset = combinedDataset

        # total_samples = len(combinedDataset)
        # print(f"Total Batched samples after shuffle: {total_samples}")
        Dataset.print_dataset_info("Combined dataset", combinedDataset)
        newInstance.dataset = combinedDataset
        # newInstance.saveCachedDataset(csvList)
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
        print(f"Dataset: {name}", end="")
        iterator = iter(dataset)
        count = tf.data.experimental.cardinality(dataset).numpy()
        # count = 1
        if isinstance(dataset, tf.data.Dataset):
            # Inspect a single batch
            example = next(iterator)
            # print("Next", len(example))
            x, y = example  # Take one batch
            print(
                f"\tType: TF Dataset\t length: {count}\tinputShape: {x.shape}\ttargetShape: {y.shape}\tinputType: {x.dtype}\ttargetType: {y.dtype}"
            )
        # Check if dataset is a NumPy array
        elif isinstance(dataset, np.ndarray):
            print(
                f"Type: NumPy Array\tlength: {count}\tshape: {dataset.shape}\ttype: {dataset.dtype}\tfirst5: {dataset[:5]}"
            )
        # Check if dataset is a pandas DataFrame
        elif isinstance(dataset, pd.DataFrame):
            print(
                f"Type: Pandas DataFrame\tlength: {count}\tshape: {dataset.shape}\ncolumns:\n{dataset.dtypes}\nfirst5:\n{dataset.head()}\nstats:\n{dataset.describe()}"
            )
        else:
            print("Unsupported dataset type", end="")

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
        # print(dataset)
        # self.plot_tf_dataset(dataset)
        Dataset.print_dataset_info("PreShuffle", dataset)
        # wait for 10 seconds
        # time.sleep(10)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)
        # time.sleep(10)
        Dataset.print_dataset_info("PostShuffle", dataset)

        # dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Optimize data loading
        # Print dataset elements for verification

        return dataset

    @staticmethod
    def numOfFeatures():
        return len(Dataset.acceptedFeatures)

    @staticmethod
    def plotFeatures(dataset, block=False):
        # Create a 3x2 grid of subplots showing each feature vs temperature
        features_list = []
        targets_list = []

        for features, target in dataset:
            # Iterate through the dataset to extract the features and target
            if len(features.shape) == 3:  # (batch_size, timesteps, num_features)
                timestep_0_features = features[:, 0, :]  # Select the first timestep
            else:
                timestep_0_features = features  # No timesteps, use the data as is
            features_list.append(
                timestep_0_features.numpy()
            )  # Convert to numpy for plotting
            targets_list.append(target.numpy())  # Convert to numpy for plotting
        features_array = np.vstack(
            features_list
        )  # Shape will be (batch_size * len(dataset), 5)
        targets_array = np.vstack(
            targets_list
        )  # Shape will be (batch_size * len(dataset), 1)

        # check arrays for nans
        if np.isnan(features_array).any():
            print("features_array has NaNs")
            raise ValueError("features_array has NaNs")
        if np.isnan(targets_array).any():
            print("targets_array has NaNs")
            raise ValueError("targets_array has NaNs")

        fig, axes = plt.subplots(3, 2, figsize=(10, 6))
        fig.subplots_adjust(hspace=0.5)  # Adjust spacing between subplots
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < Dataset.numOfFeatures():
                ax.set_title(Dataset.acceptedFeatures[i])
                ax.scatter(
                    targets_array, features_array[:, i], s=5, color="blue", alpha=0.7
                )

        plt.show(block=block)

        return

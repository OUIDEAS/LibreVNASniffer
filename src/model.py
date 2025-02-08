import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from touchstone import Touchstone, TouchstoneList
import seaborn as sns
import functools
from enum import Enum
import matplotlib.cm as cm
from dataset import Dataset
from scaler import Scaler


EPOCHS = 750
# BUFFER_SIZE = 10000
# BATCH_SIZE = 20


class Model:
    def __init__(self) -> None:
        self.model = None
        # self.scaler_X = MinMaxScaler()
        # self.scaler_y = MinMaxScaler()
        # self.validationRatio = 0.3
        # self.epochs = EPOCHS
        # self.buffer_size = BUFFER_SIZE
        # self.batch_size = BATCH_SIZE
        self.modelName = "Default Model name"
        self.scaler = None
        # self.acceptedFeatures = [
        #     "resonanceFrequency",
        #     "resonanceMagnitude",
        #     "resonancePhase",
        #     "resonanceReal",
        #     "resonanceImag",
        # ]

    def initModel(self):
        raise NotImplementedError("Subclasses must implement this method")

    # Gets the scalars for the X and y values

    def getScaler(self):
        return self.scaler

    # sets the scalars for the X and y values
    def setScaler(self, scaler):
        self.scaler = scaler

    # # Gets the scalars for the X and y values
    # def getScalers(self):
    #     return self.scaler_X, self.scaler_y

    # # sets the scalars for the X and y values
    # def setScalers(self, scaler_X, scaler_y):
    #     self.scaler_X = scaler_X
    #     self.scaler_y = scaler_y

    # def featuresFromTouchstone(self, touchstone: TouchstoneList):
    #     feature_map = {
    #         "resonanceFrequency": touchstone.getResonanceFrequencyList(),
    #         "resonanceMagnitude": touchstone.getResonanceMagnitudeList(),
    #         "resonancePhase": touchstone.getPhaseDataList(),
    #         "resonanceReal": [c.real for c in touchstone.getComplexDataList()],
    #         "resonanceImag": [c.imag for c in touchstone.getComplexDataList()],
    #     }

    #     # Filter features based on acceptedFeatures
    #     selected_features = [
    #         feature_map[key] for key in self.acceptedFeatures if key in feature_map
    #     ]

    #     # Convert to NumPy array and transpose
    #     data = np.array(selected_features).T.tolist()

    #     X = data
    #     y = np.array(touchstone.getTemperatureDataList())
    #     return X, y

    def saveWeights(self):
        raise NotImplementedError("Subclasses must implement this method")

    # @staticmethod
    # def r2_score_manual(y_true, y_pred):
    #     ss_res = np.sum((y_true - y_pred) ** 2)
    #     yAvg = np.mean(y_true)
    #     ss_tot = np.sum((y_true - yAvg) ** 2)

    #     r2 = 1 - (ss_res / ss_tot)
    #     return r2

    # def trainCSV(self, csvPath: str):
    #     touchstoneList = TouchstoneList.loadTouchstoneListFromCSV(csvPath)
    #     self.trainTouchstone(touchstoneList)

    # def formatFeaturesforModel(self, X, y):
    #     raise NotImplementedError("Subclasses must implement this method")

    # def trainTouchstone(self, touchstone: TouchstoneList):
    #     if self.model is None:
    #         self.initModel()

    #     X, y = self.featuresFromTouchstone(touchstone)
    #     X_model, y_model = self.formatFeaturesforModel(X, y)
    #     # Train the model
    #     if self.model.fit is None:
    #         raise NotImplementedError("model does not have a fit method")
    #     history = self.model.fit(
    #         X_model, y_model, epochs=self.epochs, validation_split=0.2, batch_size=32
    #     )
    #     return history

    # def touchstoneToDataset(self, touchstone: TouchstoneList):
    #     X, y = self.featuresFromTouchstone(touchstone)
    #     # format features for model
    #     X_lstm, y_lstm = self.formatFeaturesforModel(X, y)
    #     dataset = tf.data.Dataset.from_tensor_slices((X_lstm, y_lstm))
    #     for sample in dataset.take(1):
    #         sample_shape = sample[0].shape
    #         if hasattr(self.model, "input_shape"):
    #             assert sample_shape == self.model.input_shape[1:], (
    #                 f"Dataset shape {sample_shape} does not match the expected input shape {self.model.input_shape[1:]}"
    #             )
    #     return dataset

    def predictTouchstone(self, touchstone: TouchstoneList):
        X, y = Dataset.featuresFromTouchstone(touchstone)
        X, y = self.formatFeaturesForModel(X, y)
        yPred = self.model.predict(X)
        yTest = y

        return self.scaler.inverseTransformPrediction(yPred, yTest)

    def formatFeaturesForModel(self, X, y):
        raise NotImplementedError("Subclasses must implement this method")

    # def finalizePrediction(self, yPred, yTest):
    #     return yPred, yTest

    def predictCSV(self, csvPath: str):
        touchstoneList = TouchstoneList.loadTouchstoneListFromCSV(csvPath)
        yPred, yTest = self.predictTouchstone(touchstoneList)
        return yPred, yTest

    # def concatenateDatasets(self, datasets):
    #     if not datasets:
    #         raise ValueError("The dataset list is empty.")

    #     concatenated_dataset = datasets[0]
    #     # Use functools.reduce to concatenate all datasets in the list
    #     for i in range(1, len(datasets)):
    #         concatenated_dataset = concatenated_dataset.concatenate(datasets[i])
    #     print(
    #         f"final concatination of {len(datasets)} datasets has {len(concatenated_dataset)} samples"
    #     )
    #     return concatenated_dataset

    # def print_dataset_info(self, name, dataset):
    #     # Check if dataset is a TensorFlow Dataset
    #     print(f"Dataset: {name}")
    #     if isinstance(dataset, tf.data.Dataset):
    #         print("\tDataset Type: TensorFlow Dataset")
    #         # Inspect a single batch
    #         for x, y in dataset.take(1):  # Take one batch
    #             print("\tShape of input data:", x.shape)
    #             print("\tShape of target data:", y.shape)
    #             print("\tData types of input:", x.dtype)
    #             print("\tData types of target:", y.dtype)
    #             # print(
    #             #     "Sample input data:", x.numpy()
    #             # )  # Convert tensor to numpy array for display
    #             # print("Sample target data:", y.numpy())
    #     # Check if dataset is a NumPy array
    #     elif isinstance(dataset, np.ndarray):
    #         print("Dataset Type: NumPy Array")
    #         print("Shape of dataset:", dataset.shape)
    #         print("Data type:", dataset.dtype)
    #         print("First 5 samples:", dataset[:5])  # Print first 5 samples
    #     # Check if dataset is a pandas DataFrame
    #     elif isinstance(dataset, pd.DataFrame):
    #         print("Dataset Type: Pandas DataFrame")
    #         print("Shape of dataset:", dataset.shape)
    #         print("Data types of columns:\n", dataset.dtypes)
    #         print("First 5 rows:\n", dataset.head())  # Print first 5 rows
    #         print(
    #             "Summary statistics:\n", dataset.describe()
    #         )  # Summary stats for numerical columns
    #     else:
    #         print("Unsupported dataset type")

    # Takes a list of csv paths and turns it into a minibatched dataset
    # def datasetFromCSVList(self, csvList):
    #     datasetList = []
    #     for csv in csvList:
    #         touchstoneList = TouchstoneList.loadTouchstoneListFromCSV(csv)
    #         dataset = self.touchstoneToDataset(touchstoneList)
    #         print(f"Csv: {csv} has {len(dataset)} samples")
    #         # print(f"Element spec of {csv}:", dataset.element_spec)
    #         datasetList.append(dataset)

    #     combinedDataset = self.concatenateDatasets(datasetList)
    #     # Check the shape of dataset elements
    #     print("Element spec:", combinedDataset.element_spec)
    #     # combinedDataset = combinedDataset.batch(1)
    #     self.print_dataset_info("PreShuffle", combinedDataset)
    #     combinedDataset = self.shuffleDataset(combinedDataset)
    #     self.print_dataset_info("PostShuffle", combinedDataset)
    #     combinedDataset = combinedDataset

    #     total_samples = len(combinedDataset)
    #     print(f"Total Batched samples after shuffle: {total_samples}")
    #     return combinedDataset

    # Splits a dataset into validation and training datasets
    # def splitDataset(self, dataset):
    #     # Create validation dataset
    #     total_samples = len(dataset)
    #     validation_size = int(self.validationRatio * total_samples)
    #     validation_dataset = dataset.take(validation_size)

    #     # Create training dataset
    #     training_dataset = dataset.skip(validation_size)
    #     return training_dataset, validation_dataset

    # Trains a model on a list of csv files
    # def miniBatchTrain(self, csvList):
    #     if self.model is None:
    #         self.initModel()
    #     if self.model.fit is None:
    #         raise NotImplementedError("model does not have a fit method")

    #     combinedDataset = self.datasetFromCSVList(csvList)
    #     training_dataset, validation_dataset = self.splitDataset(combinedDataset)

    #     history = self.trainOnDataset(training_dataset, validation_dataset)
    #     mae = self.predictOnDataset(validation_dataset)
    #     return (history, mae)

    # def shuffleDataset(self, dataset):
    #     # Shuffle and batch the dataset
    #     print(dataset)
    #     # self.plot_tf_dataset(dataset)
    #     print("Shuffling dataset")
    #     dataset = dataset.batch(self.batch_size)
    #     dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)
    #     print("Shuffled")

    #     # dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Optimize data loading
    #     # Print dataset elements for verification

    #     return dataset

    def trainOnDataset(self, dataset, split, epochs=EPOCHS):
        raise NotImplementedError("Subclasses must implement this method")

    def getTFDatasetFromDataset(self, dataset: Dataset):
        raise NotImplementedError("Subclasses must implement this method")

    # Returns the Mean Absolute Error (MAE) of the model on the validation dataset
    def predictOnDataset(self, dataset: Dataset, split):
        # split
        filteredDataset = self.getTFDatasetFromDataset(dataset)

        _, validation = Dataset.splitDataset(filteredDataset, split)
        # Calculate MAE on validation dataset
        y_true = []
        y_pred = []

        # Iterate through the validation dataset
        dataset.print_dataset_info("Validation Dataset", validation)
        for x_batch, y_batch in validation:
            # print(f"Input data shape: {x_batch.shape}")  # Should print (20, 10, 5)
            # print(f"Target data shape: {y_batch.shape}")  # Should print (20, 1)
            # print(type(x_batch))
            # print(type(y_batch))
            # self.print_dataset_info("Batch", x_batch)
            # print("Making Predictions")
            # Make predictions for the current batch
            # print(x_batch)
            y_batch_pred = self.model.predict(x_batch)

            if np.isnan(y_batch_pred).any():
                print("x_batch")
                print(x_batch)
                print(x_batch.shape)
                print("y_batch")

                print(y_batch_pred)
                print("y_batch_pred has NaNs")

            # Store the true labels and predictions
            yPredS, yTestS = self.scaler.inverseTransformPrediction(
                y_batch_pred, y_batch
            )
            y_true.append(yTestS)  # Convert from Tensor to numpy array
            y_pred.append(yPredS)  # Predictions are already in numpy array format

        # Flatten the lists to make sure we have a single array of predictions and true labels
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        print("y_true")
        print(y_true.shape)
        # print(y_true)
        print("y_pred")
        print(y_pred.shape)
        # print(y_pred)

        # plt.figure(figsize=(10, 6))
        # plt.plot(y_true, label="True Values", color="b")
        # plt.plot(y_pred, label="Predicted Values", color="r", linestyle="--")

        # # Adding labels and a legend
        # plt.xlabel("Index")
        # plt.ylabel("Value")
        # plt.title("True vs Predicted Values")
        # plt.legend()

        # # Show the plot
        # plt.show()

        # Calculate the Mean Absolute Error
        mae_tf = tf.keras.losses.MeanAbsoluteError()
        mae_value = mae_tf(y_true, y_pred).numpy()
        print(f"MAE without scaling factor: {mae_value}")

        return mae_value, y_pred, y_true

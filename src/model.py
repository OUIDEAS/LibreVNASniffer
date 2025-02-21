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
        self.modelName = "Default Model name"
        self.scaler = None

    def initModel(self):
        raise NotImplementedError("Subclasses must implement this method")

    # Gets the scalars for the X and y values

    def getScaler(self):
        return self.scaler

    # sets the scalars for the X and y values
    def setScaler(self, scaler):
        self.scaler = scaler

    def saveWeights(self):
        raise NotImplementedError("Subclasses must implement this method")

    def predictTouchstone(self, touchstone: TouchstoneList):
        X, y = Dataset.featuresFromTouchstone(touchstone)
        X, y = self.formatFeaturesForModel(X, y)
        print("Instancing model " + self.modelName + " on data with shape:", X.shape)
        yPred = self.predict(X)
        yTest = y

        if np.isnan(yPred).any():
            print("pred has NaNs")
            raise ValueError("y_pred has NaNs")

        return self.scaler.inverseTransformPrediction(yPred, yTest)

    def formatFeaturesForModel(self, X, y):
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement this method")

    def checkPredictionForNans(self, yPred):
        if np.isnan(yPred).any():
            print("pred has NaNs")
            raise ValueError("y_pred has NaNs")
        return yPred

    # def finalizePrediction(self, yPred, yTest):
    #     return yPred, yTest

    def predictCSV(self, csvPath: str):
        touchstoneList = TouchstoneList.loadTouchstoneListFromCSV(csvPath)
        yPred, yTest = self.predictTouchstone(touchstoneList)
        return yPred, yTest

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
            y_batch_pred = self.predict(x_batch)
            # Store the true labels and predictions
            yPredS, yTestS = self.scaler.inverseTransformPrediction(
                y_batch_pred, y_batch.numpy()
            )
            y_true.append(yTestS)  # Convert from Tensor to numpy array
            y_pred.append(yPredS)  # Predictions are already in numpy array format

        # Flatten the lists to make sure we have a single array of predictions and true labels
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        assert y_true.shape == y_pred.shape
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
        # print(f"MAE from validation dataset: {mae_value}")

        return mae_value, y_pred, y_true

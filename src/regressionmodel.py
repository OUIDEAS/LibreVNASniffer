import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from touchstone import Touchstone, TouchstoneList
import seaborn as sns
import functools
from enum import Enum
from model import Model
from dataset import Dataset


class RegressionModel(Model):
    def initModel(self):
        if self.model is None:
            print("Initializing Regression Model")
            self.model = LinearRegression()
            self.modelName = "Regression Model"
        return

    def saveWeights(self):
        # Save the model weights
        self.model.save_weights("regression.weights.h5")

        # Save the entire model (optional)
        self.model.save("regression.model.h5")

    def getTFDatasetFromDataset(self, dataset: Dataset):
        if dataset.timesteps >= 1:
            return Dataset.includeOnlyFeatures(
                Dataset.getTimesteplessCopy(dataset.dataset),
                ["normalizedResonanceFrequency"],
            )

        else:
            return dataset.includeOnlyFeatures(["normalizedResonanceFrequency"])

    def formatFeaturesForModel(self, X, y):
        X, y = Dataset.formatFeatures(X, y)
        X, y = self.scaler.smartScaleFeatures(X=X, y=y)
        X, y = Dataset.includeOnlyFeature(X, y, ["normalizedResonanceFrequency"])

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if not isinstance(y, np.ndarray):
            y = np.array(y)
        return X, y

    def predict(self, X):
        return self.checkPredictionForNans(self.model.predict(X))

    def trainOnDataset(self, dataset: Dataset, split, epochs):
        if self.model is None:
            self.initModel()

        onlyResFreq = self.getTFDatasetFromDataset(dataset)

        training_dataset, testing_dataset = Dataset.splitDataset(onlyResFreq, split)

        # Initialize empty lists to store the data
        X_list = []
        y_list = []

        # Iterate through the dataset and append the values to the lists
        for features, label in training_dataset:
            X_list.extend(features.numpy())  # Flatten the batch and append features
            y_list.extend(label.numpy())  # Flatten the batch and append labels
        # Train the model
        print(
            "training Regression model on Dataset with shape {}".format(
                np.array(X_list).shape
            )
        )
        history = self.model.fit(X_list, y_list)

        self.setScaler(dataset.getScaler())

        return None

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
            return dataset.getTimesteplessCopy()
        else:
            return dataset.dataset

    def formatFeaturesForModel(self, X, y):
        X, y = Dataset.formatFeatures(X, y)
        X, _ = self.scaler.fitAndScaleFeatures(X=X, y=None)
        _, y = self.scaler.scaleFeatures(X=None, y=y)
        return X, y

    def trainOnDataset(self, dataset: Dataset, split, epochs):
        if self.model is None:
            self.initModel()

        noTimesteps = dataset.getTimesteplessCopy()

        training_dataset, testing_dataset = Dataset.splitDataset(noTimesteps, split)

        # Initialize empty lists to store the data
        X_list = []
        y_list = []

        # Iterate through the dataset and append the values to the lists
        for features, label in training_dataset:
            X_list.extend(features.numpy())  # Flatten the batch and append features
            y_list.extend(label.numpy())  # Flatten the batch and append labels
        # Train the model
        history = self.model.fit(X_list, y_list)

        self.setScaler(dataset.getScaler())

        return None

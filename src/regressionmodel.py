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

    def formatFeaturesforModel(self, X, y):
        X_scaled = self.scaler_X.fit_transform(X)

        # Optionally normalize the target (y)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))

        # Since this is a NN, we don't need to reshape the input data
        # Flatten the input for feedforward NN (timesteps * features)
        X_nn = X_scaled.reshape((X_scaled.shape[0], -1))  # Flatten

        return X_nn, y_scaled

    def finalizePrediction(self, yPred, yTest):
        yPred, yTest = super().finalizePrediction(yPred, yTest)
        # Takes off the amout of data related to the timesteps
        yPred = yPred[self.timesteps :]
        yTest = yTest[self.timesteps :]
        return (yPred, yTest)

    def trainOnDataset(self, training_dataset, validation_dataset):
        if self.model is None:
            self.initModel()
        # Initialize empty lists to store the data
        X_list = []
        y_list = []

        # Iterate through the dataset and append the values to the lists
        for features, label in training_dataset:
            X_list.extend(features.numpy())  # Flatten the batch and append features
            y_list.extend(label.numpy())  # Flatten the batch and append labels
        # Train the model
        history = self.model.fit(X_list, y_list)
        return history

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from touchstone import Touchstone, TouchstoneList
import seaborn as sns
import functools
from enum import Enum
from model import Model


class NNModel(Model):
    def initModel(self):
        if self.model is None:
            print("Initializing Feedforward Neural Network Model")
            # Check if GPU is available
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                print(f"GPUs available: {gpus}")
            else:
                print("No GPUs found.")

            self.model = Sequential(
                [
                    Input(shape=(5,)),  # Flatten input features (timesteps * features)
                    Dense(
                        64, activation="relu", kernel_regularizer=regularizers.l2(0.001)
                    ),
                    Dropout(0.2),  # Dropout layer to avoid overfitting
                    Dense(
                        64, activation="relu", kernel_regularizer=regularizers.l2(0.001)
                    ),
                    Dropout(0.2),
                    Dense(1),  # Output layer for regression
                ]
            )

            self.model.compile(
                optimizer="adam", loss="mean_squared_error", metrics=["mae"]
            )
            self.modelName = "Feedforward Neural Network Model"

    def saveWeights(self):
        # Save the model weights
        self.model.save_weights("nn.weights.h5")

        # Save the entire model (optional)
        self.model.save("nn.model.h5")

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

        # Train the model
        history = self.model.fit(
            training_dataset, epochs=self.epochs, validation_data=validation_dataset
        )
        return history

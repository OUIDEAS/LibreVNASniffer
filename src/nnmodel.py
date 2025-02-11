import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from touchstone import Touchstone, TouchstoneList
import seaborn as sns
import functools
from enum import Enum
from model import Model
from dataset import Dataset

BASELAYERSIZE = 5000
PATIENCE = 500


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

        training_dataset, validation_dataset = Dataset.splitDataset(noTimesteps, split)
        monitor = EarlyStopping(
            monitor="val_mae",
            min_delta=1e-3,
            patience=PATIENCE,
            verbose=0,
            mode="auto",
            restore_best_weights=True,
        )

        # Train the model
        history = self.model.fit(
            training_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks=[monitor],
            verbose=0,
        )

        self.setScaler(dataset.getScaler())

        return history

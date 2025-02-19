import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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

BASELAYERSIZE = 5000
PATIENCE = 500


class LSTMModel(Model):
    def __init__(self):
        super().__init__()

    def initModel(
        self,
        timesteps=10,
        learning_rate=0.001,
        neuronPct=0.01,
        neuronShrink=1,
        kernel_regularizer=0.001,
        dropout=0.00,
        numOfFeatures=5,
    ):
        if self.model is None:
            self.timesteps = int(round(timesteps))
            print("Initializing LSTM Model")
            # Check if GPU is available
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                print(f"GPUs available: {gpus}")
            else:
                print("No GPUs found.")

            self.model = Sequential(
                [
                    Input(
                        shape=(self.timesteps, numOfFeatures)
                    ),  # Define input shape here
                    LSTM(
                        round(neuronPct * BASELAYERSIZE),
                        activation="tanh",
                        kernel_regularizer=regularizers.l2(kernel_regularizer),
                        return_sequences=True,
                    ),
                    Dropout(dropout),  # Drop 20% of the units
                    LSTM(
                        round(neuronPct * BASELAYERSIZE * neuronShrink),
                        return_sequences=False,
                    ),
                    Dropout(dropout),
                    Dense(1),
                ]
            )

            self.model.compile(
                optimizer=Adam(
                    learning_rate=learning_rate
                ),  # Explicitly set learning rate
                loss="mean_squared_error",
                metrics=["mae"],
            )
            self.modelName = "LSTM Model"
            print("Model initialized")
            print(self.model.summary())

    def saveWeights(self):
        # Save the model weights
        self.model.save_weights("lstm.weights.h5")

        # Save the entire model (optional)
        self.model.save("lstm.model.h5")

    def formatFeaturesForModel(self, X, y):
        X, y = Dataset.formatFeatures(X, y)
        X, _ = self.scaler.fitAndScaleFeatures(X=X, y=None)
        _, y = self.scaler.scaleFeatures(X=None, y=y)
        X, y = Dataset.timestepFeatures(self.timesteps, X, y)
        return X, y

    def getTFDatasetFromDataset(self, dataset: Dataset):
        if dataset.timesteps == self.timesteps:
            return dataset.dataset
        else:
            raise ValueError("Dataset timesteps do not match model timesteps")

    def trainOnDataset(self, dataset: Dataset, split, epochs):
        # Split the dataset
        training_dataset, validation_dataset = Dataset.splitDataset(
            dataset.dataset, split
        )

        AUTOTUNE = tf.data.AUTOTUNE

        training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

        if self.model is None:
            self.initModel()
        monitor = EarlyStopping(
            monitor="val_mae",
            min_delta=1e-3,
            patience=PATIENCE,
            verbose=0,
            mode="auto",
            restore_best_weights=True,
        )
        # Train the model
        print("(LSTM) Training on dataset for ", epochs, " epochs")
        history = self.model.fit(
            training_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks=[monitor],
            verbose=0,
        )

        # Track the best epoch during training
        best_val_mae = float("inf")
        best_epoch = 0
        for epoch, val_mae in enumerate(history.history["val_mae"]):
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch

        stoppedepoch = monitor.stopped_epoch
        print(f"(LSTM) Best epoch: {best_epoch}")
        print(f"(LSTM) Stopped at epoch {stoppedepoch}")

        self.setScaler(dataset.getScaler())

        return history

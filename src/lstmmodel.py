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

BASELAYERSIZE = 5000
PATIENCE = 500


class LSTMModel(Model):
    def __init__(self, timesteps=10):
        super().__init__()
        self.timesteps = timesteps

    def initModel(
        self,
        timesteps=10,
        learning_rate=0.001,
        neuronPct=0.01,
        neuronShrink=1,
        kernel_regularizer=0.001,
        dropout=0.00,
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
                    Input(shape=(self.timesteps, 5)),  # Define input shape here
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

    def formatFeaturesforModel(self, X, y):
        X_scaled = self.scaler_X.fit_transform(X)

        # Optionally normalize the target (y)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        # Reshape input data to 3D for LSTM (samples, timesteps, features)
        X_lstm = []
        y_lstm = []

        for i in range(self.timesteps, len(X_scaled)):
            X_lstm.append(X_scaled[i - self.timesteps : i])
            y_lstm.append(y_scaled[i])

        X_model = np.array(X_lstm)
        y_model = np.array(y_lstm)

        return X_model, y_model

    def finalizePrediction(self, yPred, yTest):
        yPred, yTest = super().finalizePrediction(yPred, yTest)
        return (yPred, yTest)

    def trainOnDataset(self, training_dataset, validation_dataset):
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
        print("(LSTM) Training on dataset for ", self.epochs, " epochs")
        history = self.model.fit(
            training_dataset,
            epochs=self.epochs,
            validation_data=validation_dataset,
            callbacks=[monitor],
            verbose=0,
        )
        epochs = monitor.stopped_epoch
        print(f"(LSTM) Stopped at epoch {epochs}")
        return history

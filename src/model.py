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
import matplotlib.cm as cm


class Model:
    def __init__(self, timesteps=10) -> None:
        self.model = None
        self.timesteps = timesteps
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.validationRatio = 0.2
        self.epochs = 500
        self.buffer_size = 10000
        self.batch_size = 20
        self.modelName = "Default Model name"

    def initModel(self):
        raise NotImplementedError("Subclasses must implement this method")

    def featuresFromTouchstone(self, touchstone: TouchstoneList):
        resonanceFrequency = touchstone.getResonanceFrequencyList()
        resonanceMagnitude = touchstone.getResonanceMagnitudeList()
        resonancePhase = touchstone.getPhaseDataList()
        resonanceComplex = touchstone.getComplexDataList()
        resonanceReal = [c.real for c in resonanceComplex]
        resonanceImag = [c.imag for c in resonanceComplex]
        data = np.array(
            [
                resonanceFrequency,
                resonanceMagnitude,
                resonancePhase,
                resonanceReal,
                resonanceImag,
            ]
        ).T
        data = data.tolist()
        X = data
        y = np.array(touchstone.getTemperatureDataList())
        return X, y

    def saveWeights(self):
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def r2_score_manual(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        yAvg = np.mean(y_true)
        ss_tot = np.sum((y_true - yAvg) ** 2)

        r2 = 1 - (ss_res / ss_tot)
        return r2

    def trainCSV(self, csvPath: str):
        touchstoneList = TouchstoneList.loadTouchstoneListFromCSV(csvPath)
        self.trainTouchstone(touchstoneList)

    def formatFeaturesforModel(self, X, y):
        raise NotImplementedError("Subclasses must implement this method")

    def trainTouchstone(self, touchstone: TouchstoneList):
        if self.model is None:
            self.initModel()

        X, y = self.featuresFromTouchstone(touchstone)
        X_model, y_model = self.formatFeaturesforModel(X, y)
        # Train the model
        if self.model.fit is None:
            raise NotImplementedError("model does not have a fit method")
        history = self.model.fit(
            X_model, y_model, epochs=self.epochs, validation_split=0.2, batch_size=32
        )
        return history

    def touchstoneToDataset(self, touchstone: TouchstoneList):
        X, y = self.featuresFromTouchstone(touchstone)
        # format features for model
        X_lstm, y_lstm = self.formatFeaturesforModel(X, y)
        dataset = tf.data.Dataset.from_tensor_slices((X_lstm, y_lstm))
        for sample in dataset.take(1):
            sample_shape = sample[0].shape
            if hasattr(self.model, "input_shape"):
                assert (
                    sample_shape == self.model.input_shape[1:]
                ), f"Dataset shape {sample_shape} does not match the expected input shape {self.model.input_shape[1:]}"
        return dataset

    def predictTouchstone(self, touchstone: TouchstoneList):
        X, y = self.featuresFromTouchstone(touchstone)
        X_model, y_model = self.formatFeaturesforModel(X, y)
        yPred = self.model.predict(X_model)
        yTest = y_model

        return self.finalizePrediction(yPred, yTest)

    def finalizePrediction(self, yPred, yTest):
        yPred = self.scaler_y.inverse_transform(yPred)
        yTest = self.scaler_y.inverse_transform(yTest)
        return yPred, yTest

    def predictCSV(self, csvPath: str):
        touchstoneList = TouchstoneList.loadTouchstoneListFromCSV(csvPath)
        yPred, yTest = self.predictTouchstone(touchstoneList)
        return yPred, yTest

    def concatenateDatasets(self, datasets):
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

    # Takes a list of csv paths and trains the model on each csv file

    def miniBatchTrain(self, csvList):
        if self.model is None:
            self.initModel()
        if self.model.fit is None:
            raise NotImplementedError("model does not have a fit method")

        datasetList = []
        for csv in csvList:
            touchstoneList = TouchstoneList.loadTouchstoneListFromCSV(csv)
            dataset = self.touchstoneToDataset(touchstoneList)
            print(f"Csv: {csv} has {len(dataset)} samples")
            print(f"Element spec of {csv}:", dataset.element_spec)
            datasetList.append(dataset)

        combinedDataset = self.concatenateDatasets(datasetList)
        # Check the shape of dataset elements
        print("Element spec:", combinedDataset.element_spec)
        # combinedDataset = combinedDataset.batch(1)
        total_samples = len(combinedDataset)
        print(f"Total samples before shuffle: {total_samples}")
        combinedDataset = self.shuffleDataset(combinedDataset)
        combinedDataset = combinedDataset

        total_samples = len(combinedDataset)
        print(f"Total Batched samples after shuffle: {total_samples}")

        # Create validation dataset
        validation_size = int(0.2 * total_samples)
        validation_dataset = combinedDataset.take(validation_size)

        # Create training dataset
        training_dataset = combinedDataset.skip(validation_size)

        history = self.trainOnDataset(training_dataset, validation_dataset)
        return history

    def shuffleDataset(self, dataset):
        # Shuffle and batch the dataset
        print(dataset)
        # self.plot_tf_dataset(dataset)
        print("Shuffling dataset")
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)
        print("Shuffled")

        # dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Optimize data loading
        # Print dataset elements for verification

        return dataset

    def plot_learning_curves(self, history):
        # Set up the figure
        plt.figure(figsize=(18, 5))

        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.plot(history.history["loss"], label="Training Loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Mean Absolute Error (MAE) plot
        plt.subplot(1, 3, 2)
        plt.plot(history.history["val_mae"], label="Validation MAE")
        plt.plot(history.history["mae"], label="Training MAE")
        plt.title("Mean Absolute Error (MAE)")
        plt.xlabel("Epochs")
        plt.ylabel("MAE")
        plt.legend()

        # Accuracy plot (if applicable)
        if "accuracy" in history.history:
            plt.subplot(1, 3, 3)
            plt.plot(history.history["accuracy"], label="Training Accuracy")
            plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
            plt.title("Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()

        # Show the plots
        plt.tight_layout()
        plt.show()

    def trainOnDataset(self, dataset, validation_dataset):
        raise NotImplementedError("Subclasses must implement this method")

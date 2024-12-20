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


class ModelType(Enum):
    NN = 1
    LSTM = 2
    REGRESSION = 3


EPOCHS = 500


class RNNModel:
    def __init__(self, timesteps=10) -> None:
        self.LSTMModel = None
        self.regressionModel = None
        self.NNModel = None
        self.timesteps = timesteps
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def initModel(self):
        if self.LSTMModel is None:
            print("Initializing LSTM Model")
            # Check if GPU is available
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                print(f"GPUs available: {gpus}")
            else:
                print("No GPUs found.")

            self.LSTMModel = Sequential(
                [
                    Input(shape=(self.timesteps, 5)),  # Define input shape here
                    LSTM(
                        50,
                        activation="relu",
                        kernel_regularizer=regularizers.l2(0.001),
                        return_sequences=True,
                    ),
                    # Dropout(0.2),  # Drop 20% of the units
                    LSTM(50, return_sequences=False),
                    # Dropout(0.2),
                    Dense(1),
                ]
            )

            self.LSTMModel.compile(
                optimizer="adam", loss="mean_squared_error", metrics=["mae"]
            )

        if self.NNModel is None:
            print("Initializing NN Model")
            # Check if GPU is available
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                print(f"GPUs available: {gpus}")
            else:
                print("No GPUs found.")

            self.NNModel = Sequential(
                [
                    Input(
                        shape=(self.timesteps, 5)
                    ),  # Define input shape (timesteps, features)
                    # Flatten the input to feed it into dense layers
                    tf.keras.layers.Flatten(),
                    # First Dense layer with 50 units and ReLU activation
                    Dense(
                        50, activation="relu", kernel_regularizer=regularizers.l2(0.001)
                    ),
                    # Optional: Dropout layer to prevent overfitting (uncomment if needed)
                    # Dropout(0.2),
                    # Second Dense layer with 50 units and ReLU activation
                    Dense(50, activation="relu"),
                    # Optional: Dropout layer to prevent overfitting (uncomment if needed)
                    # Dropout(0.2),
                    # Output Dense layer with a single neuron for the output
                    Dense(1),
                ]
            )

            self.NNModel.compile(
                optimizer="adam", loss="mean_squared_error", metrics=["mae"]
            )
        if self.regressionModel is None:
            print("Initializing Regression Model")
            self.regressionModel = LinearRegression()
        return

    def lstmDataFromTouchstone(self, touchstone: TouchstoneList):
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

    # def check(self, touchstone: TouchstoneList) -> None:
    #     if self.model is None:
    #         self.initModel()

    #     X, y = self.lstmDataFromTouchstone(touchstone)
    #     # y = data.iloc[:, 2].values  # Real temperature
    #     # Normalize the features
    #     X_scaled = self.scaler_X.fit_transform(X)

    #     # Optionally normalize the target (y)
    #     y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
    #     # Reshape input data to 3D for LSTM (samples, timesteps, features)
    #     X_lstm = []
    #     y_lstm = []

    #     for i in range(self.timesteps, len(X_scaled)):
    #         X_lstm.append(X_scaled[i - self.timesteps : i])
    #         y_lstm.append(y_scaled[i])

    #     X_lstm = np.array(X_lstm)
    #     y_lstm = np.array(y_lstm)

    #     # Split the data into training and testing sets
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X_lstm, y_lstm, test_size=0.2, random_state=42, shuffle=True
    #     )
    #     # Train the model
    #     history = self.model.fit(
    #         X_train, y_train, epochs=100, validation_split=0.2, batch_size=32
    #     )
    #     # Evaluate the model
    #     test_loss, test_mae = self.model.evaluate(X_test, y_test)
    #     print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
    #     # Make predictions
    #     y_pred = self.model.predict(X_test)

    #     # Optionally, inverse transform predictions and actual values for interpretation
    #     y_pred_rescaled = self.scaler_y.inverse_transform(y_pred)
    #     y_test_rescaled = self.scaler_y.inverse_transform(y_test)
    #     self.plot(y_pred_rescaled, y_test_rescaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def saveWeights(self):
        # Save the model weights
        self.LSTMModel.save_weights("lstm.weights.h5")

        # Save the entire model (optional)
        self.LSTMModel.save("lstm.model.h5")

    def r2_score_manual(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        yAvg = np.mean(y_true)
        ss_tot = np.sum((y_true - yAvg) ** 2)

        r2 = 1 - (ss_res / ss_tot)
        return r2

    def plot(self, yPred, yTest, yPredReg=None, yTestReg=None):
        # Calculate metrics
        variance = np.var(yPred)
        r2 = self.r2_score_manual(yTest, yPred)
        mae = mean_absolute_error(yTest, yPred)

        # Create a figure with two subplots (1 row, 2 columns)
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # Subplot 1: Scatter plot of Actual vs Predicted Temperatures
        ax[0].scatter(yTest, yPred, color="green")
        ax[0].set_xlabel("Actual Temperatures")
        ax[0].set_ylabel("Predicted Temperatures")
        ax[0].set_title("Actual vs Predicted Temperatures")
        ax[0].text(
            0.05,
            0.9,
            f"Variance: {variance:.2f}\nR²: {r2:.2f}\nMAE: {mae:.2f}",
            transform=ax[0].transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Subplot 2: Line plot of Actual and Predicted Temperatures
        if yPredReg is not None and yTestReg is not None:
            ax[1].plot(yTestReg, label="Actual Temperature (Regression)", color="blue")
            ax[1].plot(
                yPredReg, label="Predicted Temperature (Regression)", color="green"
            )
        ax[1].plot(yTest, label="Actual Temperature", color="blue")
        ax[1].plot(yPred, label="Predicted Temperature", color="red")
        ax[1].set_title("Predicted vs Actual Temperature")
        ax[1].set_xlabel("Sample Index")
        ax[1].set_ylabel("Temperature")
        ax[1].legend()
        if yPredReg is not None and yTestReg is not None:
            varianceReg = np.var(yPredReg)
            r2Reg = self.r2_score_manual(yTestReg, yPredReg)
            maeReg = mean_absolute_error(yTestReg, yPredReg)
            ax[1].text(
                0.05,
                0.8,
                f"Variance: {varianceReg:.2f}\nR²: {r2Reg:.2f}\nMAE: {maeReg:.2f}",
                transform=ax[1].transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.5),
            )
        ax[1].text(
            0.05,
            0.9,
            f"Variance: {variance:.2f}\nR²: {r2:.2f}\nMAE: {mae:.2f}",
            transform=ax[1].transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Adjust layout and show the combined plot
        plt.tight_layout()
        plt.show()

    def trainCSV(self, csvPath: str):
        touchstoneList = TouchstoneList.loadTouchstoneListFromCSV(csvPath)
        self.trainTouchstone(touchstoneList)

    def trainTouchstone(self, touchstone: TouchstoneList):
        if self.LSTMModel is None:
            self.initModel()

        X, y = self.lstmDataFromTouchstone(touchstone)
        # y = data.iloc[:, 2].values  # Real temperature
        # Normalize the features
        X_scaled = self.scaler_X.fit_transform(X)

        # Optionally normalize the target (y)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        # Reshape input data to 3D for LSTM (samples, timesteps, features)
        X_lstm = []
        y_lstm = []

        for i in range(self.timesteps, len(X_scaled)):
            X_lstm.append(X_scaled[i - self.timesteps : i])
            y_lstm.append(y_scaled[i])

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)

        # Split the data into training and testing sets

        # Train the model
        history = self.LSTMModel.fit(
            X_lstm, y_lstm, epochs=100, validation_split=0.2, batch_size=32
        )

    def touchstoneToDataset(self, touchstone: TouchstoneList):
        X, y = self.lstmDataFromTouchstone(touchstone)
        X, y = self.lstmDataFromTouchstone(touchstone)
        # y = data.iloc[:, 2].values  # Real temperature
        # Normalize the features
        X_scaled = self.scaler_X.fit_transform(X)

        # Optionally normalize the target (y)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        # Reshape input data to 3D for LSTM (samples, timesteps, features)
        X_lstm = []
        y_lstm = []

        for i in range(self.timesteps, len(X_scaled)):
            X_lstm.append(X_scaled[i - self.timesteps : i])
            y_lstm.append(y_scaled[i])

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)
        print(X_lstm.shape)
        dataset = tf.data.Dataset.from_tensor_slices((X_lstm, y_lstm))
        return dataset

    def predictTouchstone(self, touchstone: TouchstoneList):
        X, y = self.lstmDataFromTouchstone(touchstone)
        # y = data.iloc[:, 2].values  # Real temperature
        # Normalize the features
        X_scaled = self.scaler_X.fit_transform(X)

        # Optionally normalize the target (y)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        # Reshape input data to 3D for LSTM (samples, timesteps, features)
        X_lstm = []
        y_lstm = []

        for i in range(self.timesteps, len(X_scaled)):
            X_lstm.append(X_scaled[i - self.timesteps : i])
            y_lstm.append(y_scaled[i])

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)

        yPred = self.LSTMModel.predict(X_lstm)
        yTest = y_lstm
        # Optionally, inverse transform predictions and actual values for interpretation
        yPred = self.scaler_y.inverse_transform(yPred)
        yTest = self.scaler_y.inverse_transform(yTest)
        return yPred, yTest

    def predictTouchstoneRegression(self, touchstone: TouchstoneList):
        # Step 1: Prepare data
        X, y = self.lstmDataFromTouchstone(touchstone)

        # Normalize the features
        X_scaled = self.scaler_X.fit_transform(X)

        # Optionally normalize the target (y)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Step 2: Train the linear regression model
        # Initialize the Linear Regression model
        linear_model = self.regressionModel

        # Fit the model on the full dataset (no time-step reshaping required for linear regression)
        linear_model.fit(X_scaled, y_scaled)

        # Step 3: Make predictions
        # Predict using the same features
        yPred_scaled = linear_model.predict(X_scaled)
        # Step 4: Remove the first 'timesteps' values to align with LSTM
        yPred_aligned = yPred_scaled[self.timesteps :]
        yTest_aligned = y_scaled[self.timesteps :]

        # Optionally, inverse transform predictions and actual values for interpretation
        yPred = self.scaler_y.inverse_transform(yPred_aligned.reshape(-1, 1))
        yTest = self.scaler_y.inverse_transform(yTest_aligned.reshape(-1, 1))

        return yPred, yTest

    def predictCSV(self, csvPath: str, regression=False):
        touchstoneList = TouchstoneList.loadTouchstoneListFromCSV(csvPath)
        if regression:
            yPred, yTest = self.predictTouchstoneRegression(touchstoneList)

        else:
            yPred, yTest = self.predictTouchstone(touchstoneList)
        return yPred, yTest

    def miniBatchTrain(self, csvList, modelType):
        if self.LSTMModel is None:
            self.initModel()
        models = {
            ModelType.NN: self.NNModel,
            ModelType.LSTM: self.LSTMModel,
            ModelType.REGRESSION: self.regressionModel,
        }
        model = models[modelType]
        # for each csv file in the list, convert to touchstoneList, then dataset
        # then add to the dataset list
        datasetList = []
        for csv in csvList:
            touchstoneList = TouchstoneList.loadTouchstoneListFromCSV(csv)
            dataset = self.touchstoneToDataset(touchstoneList)
            print("Element spec:", dataset.element_spec)
            print(f"Csv: {csv} has {sum(1 for _ in dataset)} samples")
            datasetList.append(dataset)

        def concatenate_datasets(datasets):
            if not datasets:
                raise ValueError("The dataset list is empty.")

            concatenated_dataset = datasets[0]
            # Use functools.reduce to concatenate all datasets in the list
            for i in range(1, len(datasets)):
                print(f"{i} has {sum(1 for _ in concatenated_dataset)} samples")
                concatenated_dataset = concatenated_dataset.concatenate(datasets[i])
            print(f"final has {sum(1 for _ in concatenated_dataset)} samples")
            return concatenated_dataset

        combinedDataset = concatenate_datasets(datasetList)
        # Check the shape of dataset elements
        print("Element spec:", combinedDataset.element_spec)
        # combinedDataset = combinedDataset.batch(1)
        total_samples = sum(1 for _ in combinedDataset)
        print(f"Total samples before shuffle: {total_samples}")
        combinedDataset = self.shuffleDataset(combinedDataset)
        print(combinedDataset)
        combinedDataset = combinedDataset

        total_samples = sum(1 for _ in combinedDataset)
        print(f"Total samples: {total_samples}")

        # Create validation dataset
        validation_size = int(0.2 * total_samples)
        validation_dataset = combinedDataset.take(validation_size)

        # Create training dataset
        training_dataset = combinedDataset.skip(validation_size)
        print(modelType)
        print(type(model))
        print(type(self.LSTMModel))
        print(type(self.NNModel))
        print(type(self.regressionModel))

        history = model.fit(
            combinedDataset,
            validation_data=validation_dataset,
            epochs=EPOCHS,
        )
        return history

    def shuffleDataset(self, dataset):
        # Shuffle and batch the dataset
        buffer_size = 10000  # Number of chunks to shuffle
        batch_size = 20
        print(dataset)
        # self.plot_tf_dataset(dataset)
        print("Shuffling dataset")
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
        print("Shuffled")

        # dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Optimize data loading
        # Print dataset elements for verification

        return dataset

    def plot_tf_dataset(
        self, dataset, feature_names=None, target_name="Target", feature_index=0
    ):
        """
        Plots the dataset including its features and target from a tf.data.Dataset object.

        Parameters:
        - dataset: tf.data.Dataset
          The TensorFlow dataset object.
        - feature_names: list of str, optional
          List of feature names to use for labels. If None, uses default column names.
        - target_name: str, optional
          Name of the target variable for labeling purposes.

        Returns:
        - None
        """

        # Initialize lists to hold the features and target
        features_list = []
        target_list = []

        # Iterate over the dataset and collect features and target
        for features, target in dataset:
            features_list.append(features.numpy())
            target_list.append(target.numpy())

        # Convert lists to numpy arrays
        features = np.concatenate(features_list, axis=0)
        target = np.concatenate(target_list, axis=0)

        # Flatten the features if they are 3D and select only the specified feature index
        if features.ndim > 2:
            features = features.reshape(-1, features.shape[-1])

        # Select only the feature at the specified index
        if features.shape[1] > feature_index:
            features = features[:, feature_index]
        else:
            raise IndexError(
                f"Feature index {feature_index} is out of range for features with shape {features.shape}"
            )

        # Convert to DataFrame
        df = pd.DataFrame(
            {feature_names[feature_index] if feature_names else "Feature": features}
        )
        df[target_name] = target

        # Create scatter plot for the selected feature vs. target
        plt.figure(figsize=(8, 6))
        plt.scatter(df[df.columns[0]], df[target_name], alpha=0.5)
        plt.title(f"{df.columns[0]} vs. {target_name}")
        plt.xlabel(df.columns[0])
        plt.ylabel(target_name)
        plt.show()

        # Optionally plot pairplot if only one feature is used
        if feature_names is None or len(feature_names) == 1:
            sns.pairplot(df, hue=target_name, palette="viridis")
            plt.show()

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


# Below are a list of csv paths inside a list variable we use to train from multiple csv files
trainPaths = [
    "./data/run-20240815_152530/csv_20240815_152530.csv",
    "./data/run-20240815_145106/csv_20240815_145106.csv",
    "./data/run-20240712_160241/csv_20240712_160241.csv",
    "./data/run-20240710_154651/csv_20240710_154651.csv",
]


print("Hello World")
model = RNNModel(10)
history = model.miniBatchTrain(trainPaths, ModelType.LSTM)
history2 = model.miniBatchTrain(trainPaths, ModelType.NN)
model.plot_learning_curves(history)
# for path in trainPaths:
#     model.trainCSV(path)

# print("Ready to predict")
# input()


for path in trainPaths:
    yPredReg, yTestReg = model.predictCSV(path, regression=True)
    yPred, yTest = model.predictCSV(path)
    model.plot(yPred, yTest, yPredReg, yTestReg)

# model.plot()

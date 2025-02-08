# Model plotter class
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import mean_absolute_error
from model import Model
from dataset import Dataset
from sklearn.metrics import r2_score
import tensorflow as tf


class ModelPlotter:
    @staticmethod
    def plotLearningCurves(history, scaler):
        if history is None:
            return
        # Set up the figure
        plt.figure(figsize=(18, 5))
        plt.grid(True)
        scale_factor = scaler.scaler_y.data_max_ - scaler.scaler_y.data_min_
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.plot(history.history["loss"], label="Training Loss")
        plt.grid(True)
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Mean Absolute Error (MAE) plot
        historyOriginalMAE = history.history["mae"] * scale_factor
        historyOriginalValMAE = history.history["val_mae"] * scale_factor
        plt.subplot(1, 3, 2)
        plt.plot(historyOriginalValMAE, label="Validation MAE")
        plt.plot(historyOriginalMAE, label="Training MAE")
        plt.grid(True)
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
        plt.grid(True)
        plt.show()

    @staticmethod
    def predVsTrue(predictions_and_tests):
        plt.figure(figsize=(8, 6))

        for idx, (name, yPred, yTest) in enumerate(predictions_and_tests):
            # Calculate R² score
            r2 = r2_score(yTest.flatten(), yPred.flatten())

            # Calculate MAE
            mae_tf = tf.keras.losses.MeanAbsoluteError()
            mae_value = mae_tf(yTest, yPred).numpy()

            # Color for each plot line
            color = plt.cm.jet(
                idx / len(predictions_and_tests)
            )  # Use a colormap for color variation
            variance = np.var(yPred)

            # Scatter plot of true vs predicted values
            plt.scatter(
                yTest,
                yPred,
                color=color,
                alpha=0.7,
                label=f"{name} - R²: {r2:.2f}, MAE: {mae_value:.2f}, Variance: {variance:.2f}",
            )

            # Line of 1:1 correlation (slope = 1, intercept = 0)
            plt.plot([min(yTest), max(yTest)], [min(yTest), max(yTest)], color="red")

        # Add labels and title
        plt.xlabel("True Temperatures (Celsius)")
        plt.ylabel("Predicted Temperatures (Celsius)")
        plt.title("True vs Predicted Values")

        # Show the plot with legends and grid
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plotEstimateOnCSV(models, path):
        def trim_arrays(arrays):
            # Find the minimum length of the arrays
            min_length = min(len(arr) for arr in arrays)

            # Trim each array to the minimum length
            trimmed_arrays = [arr[-min_length:] for arr in arrays]

            return trimmed_arrays

        yPreds = []
        yTests = []
        for model in models:
            yPred, yTest = model.predictCSV(path)
            yPreds.append(yPred)
            yTests.append(yTest)
        # Number of models
        yPreds = trim_arrays(yPreds)
        yTests = trim_arrays(yTests)
        n_models = len(yPreds)
        assert all(np.array_equal(array, yTests[0]) for array in yTests[1:]), (
            "Not all values in the list are the same"
        )
        # Create a color map for unique colors per model
        colors = cm.get_cmap("viridis", n_models)

        # Create a figure for side-by-side subplots (1 row, 2 columns)
        fig, (ax_scatter, ax_line) = plt.subplots(1, 2, figsize=(14, 6))

        # Initialize variables to store the text info for each model
        text_info = ""

        for i, yPred in enumerate(yPreds):
            # Calculate metrics for each model
            variance = np.var(yPred)
            r2 = Dataset.r2_score_manual(
                yTests[i], yPred
            )  # Calculate R² for each model
            mae = mean_absolute_error(yTests[i], yPred)  # Calculate MAE for each model

            # Add this model's metrics to the text info
            text_info = f"{models[i].modelName} - Variance: {variance:.2f}, R²: {r2:.2f}, MAE: {mae:.2f}"

            # Scatter plot: Actual vs Predicted Temperatures for this model
            ax_scatter.scatter(
                yTests[i],
                yPred,
                label=f"Model {models[i].modelName}",
                alpha=0.7,
                color=colors(i),
            )
            ax_scatter.set_xlabel("Actual Temperatures")
            ax_scatter.set_ylabel("Predicted Temperatures")
            ax_scatter.set_title("Actual vs Predicted Temperatures")
            ax_scatter.text(
                0.05,
                0.9 - i * 0.1,
                text_info,  # Display model-specific metrics
                transform=ax_scatter.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.5),
            )

            # Line plot: Actual vs Predicted Temperatures for this model

            ax_line.plot(
                yPred,
                label=f"Predicted Temperature Model {models[i].modelName}",
                color=colors(i),
            )
            ax_line.set_title("Predicted vs Actual Temperature")
            ax_line.set_xlabel("Sample Index")
            ax_line.set_ylabel("Temperature")
        ax_line.plot(yTests[0], label="Actual Temperature", color="blue")
        # put lege
        ax_line.legend(loc="upper right")
        # Adjust layout for the combined figure
        plt.tight_layout()

        # Return the combined figure
        return fig

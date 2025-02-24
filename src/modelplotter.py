# Model plotter class
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import mean_absolute_error
from model import Model
from dataset import Dataset
from sklearn.metrics import r2_score
import tensorflow as tf
import os
from datetime import datetime


class ModelPlotter:
    curDate = datetime.now().strftime("%m-%d-%Y")

    @staticmethod
    def plotLearningCurves(history, scaler, name):
        if history is None:
            return
        # Set up the figure
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        plt.grid(True)

        yScaler = scaler.getScaler("temperature")
        scale_factor = yScaler.data_max_ - yScaler.data_min_
        # # Loss plot
        # plt.subplot(1, 3, 1)
        # plt.plot(history.history["val_loss"], label="Validation Loss")
        # plt.plot(history.history["loss"], label="Training Loss")
        # plt.grid(True)
        # plt.title("Loss")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.legend()

        # Mean Absolute Error (MAE) plot
        historyOriginalMAE = history.history["mae"] * scale_factor
        historyOriginalValMAE = history.history["val_mae"] * scale_factor
        axs.plot(historyOriginalValMAE, label="Validation MAE")
        axs.plot(historyOriginalMAE, label="Training MAE")
        axs.grid(True)
        axs.set_title("Mean Absolute Error (MAE) of " + name)
        axs.set_xlabel("Epochs")
        axs.set_ylabel("MAE")
        axs.legend()

        # Show the plots
        plt.tight_layout()
        plt.grid(True)
        # Remove spaces in name
        noSpaceName = name.replace(" ", "_")
        ModelPlotter.saveFigure(fig, noSpaceName + "_learning_curves")

        fig, axs = plt.subplots(1, 1, figsize=(6, 6))

        axs.axis("off")
        for i, text in enumerate(Dataset.acceptedFeatures):
            axs.text(
                0.05,
                0.9 - i * 0.05,
                text,
                transform=axs.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.5),
            )
        axs.text(
            0.05,
            0.9 - len(Dataset.acceptedFeatures) * 0.05,
            f"Avg of last 15 Epochs: {np.mean(historyOriginalValMAE[-15:]):.2f}",
            transform=axs.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Show the plots
        plt.tight_layout()
        plt.grid(True)
        ModelPlotter.saveFigure(fig, noSpaceName + "_features")

        # Accuracy plot (if applicable)
        # if "accuracy" in history.history:
        #     plt.subplot(1, 3, 3)
        #     plt.plot(history.history["accuracy"], label="Training Accuracy")
        #     plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        #     plt.title("Accuracy")
        #     plt.xlabel("Epochs")
        #     plt.ylabel("Accuracy")
        #     plt.legend()

        # plt.show(block=False)

    @staticmethod
    def write_to_tex_commands(
        metrics, dir=("./figures/"), filename="model_metrics.tex"
    ):
        with open(dir + ModelPlotter.curDate + "/" + filename, "w") as f:
            # Write LaTeX command for each model metric
            for model_name, r2, mae, variance in metrics:
                f.write(
                    f"\\newcommand{{\\varRScore{model_name.replace(' ', '')}}}{{{r2:.4f}}}\n"
                )
                f.write(
                    f"\\newcommand{{\\varMAE{model_name.replace(' ', '')}}}{{{mae:.4f}}}\n"
                )
                f.write(
                    f"\\newcommand{{\\varVariance{model_name.replace(' ', '')}}}{{{variance:.4f}}}\n"
                )
                f.write("\n")
        print(f"Metrics saved to {filename}")

    @staticmethod
    def predVsTrue(predictions_and_tests):
        plt.figure(figsize=(8, 6))
        metrics = []

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
                s=5,
                label=f"{name} - R²: {r2:.2f}, MAE: {mae_value:.2f}, Variance: {variance:.2f}",
            )
            metrics.append((name, r2, mae_value, variance))

            # Line of 1:1 correlation (slope = 1, intercept = 0)
            plt.plot([min(yTest), max(yTest)], [min(yTest), max(yTest)], color="red")

        # Add labels and title
        plt.xlabel("True Temperatures (Celsius)")
        plt.ylabel("Predicted Temperatures (Celsius)")
        plt.title("True vs Predicted Values")

        # Show the plot with legends and grid
        plt.legend()
        plt.grid(True)
        # plt.show(block=False)
        ModelPlotter.saveFigure(plt.gcf(), "pred_vs_true")
        ModelPlotter.write_to_tex_commands(metrics)

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
        ModelPlotter.saveFigure(fig, "estimate_on_csv")

        # Return the combined figure
        return fig

    @staticmethod
    def saveFigure(fig, figName):
        # Get current date in mm-dd-yyyy format

        # Define folder and file path
        folder_path = os.path.join("./figures", ModelPlotter.curDate)
        file_path = os.path.join(folder_path, f"{figName}.png")

        # Create folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # Save figure
        fig.savefig(file_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

        print(f"Figure saved at: {file_path}")

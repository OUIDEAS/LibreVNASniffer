# Model plotter class
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import mean_absolute_error
from model import Model


class ModelPlotter:
    @staticmethod
    def plotEstimateOnCSV(models, path):
        yPreds = []
        yTests = []
        for model in models:
            yPred, yTest = model.predictCSV(path)
            yPreds.append(yPred)
            yTests.append(yTest)
        # Number of models
        n_models = len(yPreds)
        assert all(
            np.array_equal(array, yTests[0]) for array in yTests[1:]
        ), "Not all values in the list are the same"
        # Create a color map for unique colors per model
        colors = cm.get_cmap("viridis", n_models)

        # Create a figure for side-by-side subplots (1 row, 2 columns)
        fig, (ax_scatter, ax_line) = plt.subplots(1, 2, figsize=(14, 6))

        # Initialize variables to store the text info for each model
        text_info = ""

        for i, yPred in enumerate(yPreds):
            # Calculate metrics for each model
            variance = np.var(yPred)
            r2 = Model.r2_score_manual(yTests[i], yPred)  # Calculate R² for each model
            mae = mean_absolute_error(yTests[i], yPred)  # Calculate MAE for each model

            # Add this model's metrics to the text info
            text_info = f"Model {models[i].modelName} - Variance: {variance:.2f}, R²: {r2:.2f}, MAE: {mae:.2f}"

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

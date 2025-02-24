from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


MAX_TEMP = 200
MAX_DISTANCE = 1000
MIN_TEMP = 20


class Scaler:
    def __init__(self):
        self.scaler_map = {
            "resonanceFrequency": (MinMaxScaler(), None, None),
            "normalizedResonanceFrequency": (MinMaxScaler(), MIN_TEMP, MAX_TEMP),
            # "resonanceMagnitude": (MinMaxScaler(), -100, -40),
            "resonanceMagnitude": (StandardScaler(), None, None),
            "resonancePhase": (StandardScaler(), None, None),
            "resonanceReal": (MinMaxScaler(), None, None),
            "resonanceImag": (MinMaxScaler(), None, None),
            "deltaResonanceFrequency": (MinMaxScaler(), None, None),
            "temperature": (MinMaxScaler(), MIN_TEMP, MAX_TEMP),
            "distance": (MinMaxScaler(), 0, MAX_DISTANCE),
        }

        self.fit_scalers()
        print("Scaler initialized")

    # Function to fit scalers
    def fit_scalers(self):
        for feature, (scaler, min_val, max_val) in self.scaler_map.items():
            if scaler is not None and min_val is not None and max_val is not None:
                # Generate example data assuming feature's range
                # Here we're generating an example array between min_val and max_val for each feature
                feature_data = np.array(
                    [[x] for x in np.linspace(min_val, max_val, 10)]
                )
                # Fit the scaler using the generated data
                scaler.fit(feature_data)
                print(f"Fitted {feature} scaler with range ({min_val}, {max_val})")
            else:
                print(f"{feature} has no predefined scale, skipping.")

    def smartScaleFeatures(self, X, y):
        # Print the min and max values of each X feature and y
        print(
            "Before scaling",
            f"X shape: {np.array(X).shape}",
            f"y shape: {np.array(y).shape}",
            f"y min: {np.min(y)} y max: {np.max(y)}",
        )
        # Loop through accepted features and apply scaling where necessary
        scaled_X = np.array(
            X.copy()
        )  # Make a copy of X to avoid modifying the original

        # Iterate over accepted features and apply scalers
        from dataset import Dataset

        for i, feature_name in enumerate(Dataset.acceptedFeatures):
            # Check if there is a scaler for this feature
            if self.scaler_map[feature_name] is not None:
                scaler = self.getScaler(feature_name)
                (min_val, max_val) = self.scaler_map[feature_name][1:]
                feature_column = scaled_X[:, i].reshape(
                    -1, 1
                )  # Extract the feature column

                # Apply the scaler to the feature column
                if min_val is not None and max_val is not None:
                    # If min and max values are provided, use them to scale
                    scaled_X[:, i] = scaler.transform(feature_column).flatten()
                else:
                    scaled_X[:, i] = scaler.fit_transform(feature_column).flatten()

        scaled_Y = np.array(y.copy())
        if self.getScaler("temperature") is not None:
            scaler = self.getScaler("temperature")
            scaled_Y = scaler.transform(y.reshape(-1, 1))

        print(
            "After scaling",
            f"X shape: {np.array(scaled_X).shape}",
            f"y shape: {np.array(scaled_Y).shape}",
            f"y min: {np.min(scaled_Y)} y max: {np.max(scaled_Y)}",
        )

        return scaled_X.tolist(), scaled_Y.tolist()

    # def scaleFeatures(self, X, y):
    #     X_scaled = None
    #     y_scaled = None
    #     print("Scaled Features")
    #     if X is not None:
    #         X_scaled = self.scaler_X.transform(X)
    #         print(f"X shape: {np.array(X_scaled).shape}")
    #     if y is not None:
    #         y_scaled = self.scaler_y.transform(y)
    #         print(f"y shape: {np.array(y_scaled).shape}")
    #     return X_scaled, y_scaled

    # def fitAndScaleFeatures(self, X, y):
    #     # if hasattr(self.scaler_X, "data_max_"):
    #     #     raise ValueError("Scaler is already fitted")
    #     print("Scaled and fit Features")
    #     X_scaled = None
    #     y_scaled = None
    #     if X is not None:
    #         X_scaled = self.scaler_X.fit_transform(X)
    #         print(f"X shape: {np.array(X_scaled).shape}")
    #     if y is not None:
    #         y_scaled = self.scaler_y.fit_transform(y)
    #         print(f"y shape: {np.array(y_scaled).shape}")
    #     return X_scaled, y_scaled

    # def getScalers(self):
    #     return self.scaler_X, self.scaler_y

    # # sets the scalars for the X and y values
    # def setScalers(self, scaler_X, scaler_y):
    #     self.scaler_X = scaler_X
    #     self.scaler_y = scaler_y

    def getScaler(self, featureName):
        return self.scaler_map[featureName][0]

    def inverseTransformPrediction(self, yPred, yTest):
        scaler = self.getScaler("temperature")
        # Check yPred and yTest shapes and correct types, error if not correct
        if type(yPred) != np.ndarray:
            yPred = np.array(yPred)

        if type(yTest) != np.ndarray:
            yTest = np.array(yTest)

        if len(yPred.shape) == 1:
            yPred = yPred.reshape(-1, 1)

        if len(yTest.shape) == 1:
            yTest = yTest.reshape(-1, 1)

        yPred = scaler.inverse_transform(yPred)
        yTest = scaler.inverse_transform(yTest)
        return yPred, yTest

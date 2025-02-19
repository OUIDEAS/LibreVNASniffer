from sklearn.preprocessing import MinMaxScaler
import numpy as np

MAX_TEMP = 1000


class Scaler:
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaler_map = {
            "resonanceFrequency": None,
            "normalizedResonanceFrequency": MinMaxScaler(feature_range=(0, MAX_TEMP)),
            "resonanceMagnitude": MinMaxScaler(feature_range=(-100, 0)),
            "resonancePhase": MinMaxScaler(feature_range=(-180, 180)),
            "resonanceReal": MinMaxScaler(feature_range=(-1, 1)),
            "resonanceImag": MinMaxScaler(feature_range=(-1, 1)),
            "deltaResonanceFrequency": None,
            "temperature": MinMaxScaler(feature_range=(0, MAX_TEMP)),
        }

    def smartScaleFeatures(self, acceptedFeatures, X, y):
        # Loop through accepted features and apply scaling where necessary
        scaled_X = X.copy()  # Make a copy of X to avoid modifying the original

        # Iterate over accepted features and apply scalers
        for i, feature_name in enumerate(acceptedFeatures):
            # Check if there is a scaler for this feature
            if self.scaler_map[feature_name] is not None:
                scaler = self.scaler_map[feature_name]
                feature_column = scaled_X[:, i].reshape(
                    -1, 1
                )  # Extract the feature column

                # Apply the scaler to the feature column
                scaled_X[:, i] = scaler.fit_transform(feature_column).flatten()

        scaled_Y = y.copy()
        if self.scaler_map["temperature"] is not None:
            scaler = self.scaler_map["temperature"]
            scaled_Y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

        return scaled_X, y

    def scaleFeatures(self, X, y):
        X_scaled = None
        y_scaled = None
        print("Scaled Features")
        if X is not None:
            X_scaled = self.scaler_X.transform(X)
            print(f"X shape: {np.array(X_scaled).shape}")
        if y is not None:
            y_scaled = self.scaler_y.transform(y)
            print(f"y shape: {np.array(y_scaled).shape}")
        return X_scaled, y_scaled

    def fitAndScaleFeatures(self, X, y):
        # if hasattr(self.scaler_X, "data_max_"):
        #     raise ValueError("Scaler is already fitted")
        print("Scaled and fit Features")
        X_scaled = None
        y_scaled = None
        if X is not None:
            X_scaled = self.scaler_X.fit_transform(X)
            print(f"X shape: {np.array(X_scaled).shape}")
        if y is not None:
            y_scaled = self.scaler_y.fit_transform(y)
            print(f"y shape: {np.array(y_scaled).shape}")
        return X_scaled, y_scaled

    def getScalers(self):
        return self.scaler_X, self.scaler_y

    # sets the scalars for the X and y values
    def setScalers(self, scaler_X, scaler_y):
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def inverseTransformPrediction(self, yPred, yTest):
        yPred = self.scaler_y.inverse_transform(yPred)
        yTest = self.scaler_y.inverse_transform(yTest)
        return yPred, yTest

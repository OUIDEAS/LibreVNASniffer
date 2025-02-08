from sklearn.preprocessing import MinMaxScaler
import numpy as np


class Scaler:
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

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

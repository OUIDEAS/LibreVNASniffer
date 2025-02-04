# Dataset Class

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from touchstone import Touchstone, TouchstoneList


EPOCHS = 750
BUFFER_SIZE = 10000
BATCH_SIZE = 20


class Dataset:
    def __init__(self, epochs=EPOCHS, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.validationRatio = 0.3
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.acceptedFeatures = [
            "resonanceFrequency",
            "resonanceMagnitude",
            "resonancePhase",
            "resonanceReal",
            "resonanceImag",
        ]

    def __len__(self):
        return len(self.data)

    # Gets the scalars for the X and y values
    def getScalers(self):
        return self.scaler_X, self.scaler_y

    # sets the scalars for the X and y values
    def setScalers(self, scaler_X, scaler_y):
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def featuresFromTouchstone(self, touchstone: TouchstoneList):
        feature_map = {
            "resonanceFrequency": touchstone.getResonanceFrequencyList(),
            "resonanceMagnitude": touchstone.getResonanceMagnitudeList(),
            "resonancePhase": touchstone.getPhaseDataList(),
            "resonanceReal": [c.real for c in touchstone.getComplexDataList()],
            "resonanceImag": [c.imag for c in touchstone.getComplexDataList()],
        }

        # Filter features based on acceptedFeatures
        selected_features = [
            feature_map[key] for key in self.acceptedFeatures if key in feature_map
        ]

        # Convert to NumPy array and transpose
        data = np.array(selected_features).T.tolist()

        X = data
        y = np.array(touchstone.getTemperatureDataList())
        return X, y

    @classmethod
    def from_csv(cls, path: str, target_column: str):
        df = pd.read_csv(path)
        target = df.pop(target_column)
        return cls(df, target)

    @classmethod
    def from_csv_list(cls, paths: list, target_column: str):
        data = []
        target = []
        for path in paths:
            df = pd.read_csv(path)
            target.append(df.pop(target_column))
            data.append(df)
        return cls(pd.concat(data), pd.concat(target))

    @staticmethod
    def r2_score_manual(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        yAvg = np.mean(y_true)
        ss_tot = np.sum((y_true - yAvg) ** 2)

        r2 = 1 - (ss_res / ss_tot)
        return r2

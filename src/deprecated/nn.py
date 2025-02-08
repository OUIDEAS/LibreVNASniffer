import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from sklearn.preprocessing import StandardScaler

# Assuming you have data in CSV or similar format
# Example: columns ['frequency', 'magnitude', 'phase', 'temperature']
print(os.getcwd())
data = np.loadtxt(
    "./data/run-20240815_152530/csv_20240815_152530.csv",
    delimiter=",",
    skiprows=1,
    usecols=(1, 2, 3),
)
X = data[:, 0]  # Columns 1 and 2 (ignoring the timestamp)
y = data[:, 2]  # Column 3, which is the real temperature
print(y)
X = X.reshape(-1, 1)
print(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = models.Sequential(
    [
        layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),  # Output layer for regression
    ]
)

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

model.summary()

history = model.fit(X_train, y_train, epochs=200, validation_split=0.2, batch_size=32)
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {test_mae}")
y_pred = model.predict(X_test)
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Temperatures")
plt.ylabel("Predicted Temperatures")
plt.show()

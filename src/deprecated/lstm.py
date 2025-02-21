import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the data (assuming the CSV file has columns: timestamp, freq, mag, phase, temp)
data = pd.read_csv(
    "./data/run-20240815_152530/csv_20240815_152530.csv",
    delimiter=",",
    usecols=(1, 2, 3),
    header=None,
)

# Extract features and target, ignoring timestamp
X = data.iloc[:, [0, 1]].values  # Resonance frequency, magnitude, and phase
print(X)
y = data.iloc[:, 2].values  # Real temperature

# Normalize the features
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Optionally normalize the target (y)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Reshape input data to 3D for LSTM (samples, timesteps, features)
timesteps = 10  # Number of previous timesteps to use for prediction
X_lstm = []
y_lstm = []

for i in range(timesteps, len(X_scaled)):
    X_lstm.append(X_scaled[i - timesteps : i])
    y_lstm.append(y_scaled[i])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X_lstm, y_lstm, test_size=0.2, random_state=42, shuffle=True
)
print(y_test)
print(X_train.shape[1], X_train.shape[2])

# Build the LSTM model
model = Sequential(
    [
        LSTM(50, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1),  # Output layer for temperature prediction
    ]
)

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Save the model weights
model.save_weights("lstm.weights.h5")

# Save the entire model (optional)
model.save("lstm.model.h5")

# Make predictions
y_pred = model.predict(X_test)

# Optionally, inverse transform predictions and actual values for interpretation
y_pred_rescaled = scaler_y.inverse_transform(y_pred)
y_test_rescaled = scaler_y.inverse_transform(y_test)


plt.scatter(y_test_rescaled, y_pred_rescaled)
plt.xlabel("Actual Temperatures")
plt.ylabel("Predicted Temperatures")
plt.show(block=False)

# Plot 1: Predicted vs Actual Temperature
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label="Actual Temperature", color="blue")
plt.plot(y_pred_rescaled, label="Predicted Temperature", color="red")
plt.title("Predicted vs Actual Temperature")
plt.xlabel("Sample Index")
plt.ylabel("Temperature")
plt.legend()
plt.show(block=False)

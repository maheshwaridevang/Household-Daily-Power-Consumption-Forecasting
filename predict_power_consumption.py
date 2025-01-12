import pandas as pd
import tensorflow
# Load the dataset
file_path = 'household_power_consumption.txt'  # Replace with your dataset file path
data = pd.read_csv(file_path, sep=';', parse_dates={'datetime': ['Date', 'Time']},
                   na_values=['?'], infer_datetime_format=True)

# Explore the data
print(data.head())
print(data.info())

# Drop rows with missing values
data.dropna(inplace=True)

# Convert the target feature to numeric
data['Global_active_power'] = pd.to_numeric(data['Global_active_power'])

# Aggregate data to daily consumption
data['date'] = data['datetime'].dt.date
daily_data = data.groupby('date')['Global_active_power'].sum().reset_index()
daily_data.rename(columns={'Global_active_power': 'daily_power'}, inplace=True)

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
daily_data['daily_power'] = scaler.fit_transform(daily_data[['daily_power']])

# Create sequences for LSTM
import numpy as np

def create_sequences(data, sequence_length):
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length])
    return np.array(sequences), np.array(targets)

sequence_length = 30  # Use the past 30 days to predict the next day
X, y = create_sequences(daily_data['daily_power'].values, sequence_length)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Split data into training and testing sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inverse = scaler.inverse_transform(y_pred)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
rmse = math.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(y_test_inverse, label='Actual')
plt.plot(y_pred_inverse, label='Predicted')
plt.title('Actual vs Predicted Daily Energy Consumption')
plt.xlabel('Days')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.show()

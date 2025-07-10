# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# %%
# load data
df = pd.read_csv('data//household_power_consumption.txt', sep=";", 
                 low_memory=False, na_values=['?'])
df.head()

# %%
# check the data
df.info()

# %%
# Create datetime column and handle missing values
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis=1, inplace=True)

# Convert all columns to numeric (except datetime)
for col in df.columns[:-1]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with any missing values
df.dropna(inplace=True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
# split training and test sets
# the prediction and test collections are separated over time
train = df.loc[df['datetime'] <= '2009-12-31']
test = df.loc[df['datetime'] > '2009-12-31']

# %%
# data normalization
scaler = MinMaxScaler()
cols_to_scale = [col for col in train.columns if col != 'datetime']

train_scaled = scaler.fit_transform(train[cols_to_scale])
test_scaled = scaler.transform(test[cols_to_scale])

# %%
# Prepare data for LSTM
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# We'll predict Global_active_power (column 0) using all features
time_steps = 24  # using 24 hours of historical data

X_train, y_train = create_dataset(train_scaled, train_scaled[:, 0], time_steps)
X_test, y_test = create_dataset(test_scaled, test_scaled[:, 0], time_steps)

# %%
# build a LSTM model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# %%
# train the model
early_stop = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    shuffle=False
)

# %%
# evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# %%
# plotting the predictions against the ground truth
y_pred = model.predict(X_test)

# Inverse transform the predictions and actual values
y_pred_inv = scaler.inverse_transform(
    np.concatenate((y_pred, np.zeros((len(y_pred), len(cols_to_scale)-1))), axis=1)
)[:, 0]

y_test_inv = scaler.inverse_transform(
    np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), len(cols_to_scale)-1))), axis=1)
)[:, 0]

# Plot a subset of the test predictions
plt.figure(figsize=(15, 6))
plt.plot(y_test_inv[:500], label='Actual')
plt.plot(y_pred_inv[:500], label='Predicted')
plt.title('Household Power Consumption Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Global Active Power')
plt.legend()
plt.show()
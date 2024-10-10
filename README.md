# LSTM-for-Robot-Arm-Control
Using an LSTM network to control a robot sequence of positions.
#Data Preparation
#First, we generate synthetic data representing a sequence of positions (e.g., angles for a robot arm).
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic sequence data for robot arm positions
def generate_robot_arm_data(num_sequences, sequence_length, num_features):
    data = []
    for _ in range(num_sequences):
        sequence = np.sin(np.linspace(0, 3*np.pi, sequence_length)) + np.random.normal(0, 0.1, sequence_length)
        data.append(sequence)
    data = np.array(data)
    return data.reshape(num_sequences, sequence_length, num_features)

num_sequences = 1000
sequence_length = 50
num_features = 1

data = generate_robot_arm_data(num_sequences, sequence_length, num_features)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, num_features)).reshape(num_sequences, sequence_length, num_features)

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Prepare input (X) and output (Y) sequences
def create_sequences(data, time_step):
    X, Y = [], []
    for i in range(len(data)):
        X.append(data[i, :-1])
        Y.append(data[i, 1:])
    return np.array(X), np.array(Y)

time_step = sequence_length - 1
X_train, Y_train = create_sequences(train_data, time_step)
X_test, Y_test = create_sequences(test_data, time_step)

# Build the LSTM model
#Building the LSTM Model
#Next, we build and train an LSTM model to predict the next position in the sequence.
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, num_features)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(time_step))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test), verbose=1)


# Predicting sequences
#Making Predictions
#Finally, we use the trained model to make predictions and visualize the results.
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions for plotting
train_predict = scaler.inverse_transform(train_predict.reshape(-1, num_features)).reshape(train_predict.shape)
test_predict = scaler.inverse_transform(test_predict.reshape(-1, num_features)).reshape(test_predict.shape)
Y_train = scaler.inverse_transform(Y_train.reshape(-1, num_features)).reshape(Y_train.shape)
Y_test = scaler.inverse_transform(Y_test.reshape(-1, num_features)).reshape(Y_test.shape)

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot training data predictions
plt.subplot(1, 2, 1)
plt.plot(Y_train[0], label='True Sequence')
plt.plot(train_predict[0], label='Predicted Sequence')
plt.title('Train Sequence Prediction')
plt.legend()

# Plot testing data predictions
plt.subplot(1, 2, 2)
plt.plot(Y_test[0], label='True Sequence')
plt.plot(test_predict[0], label='Predicted Sequence')
plt.title('Test Sequence Prediction')
plt.legend()

plt.show()

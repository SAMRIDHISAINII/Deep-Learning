
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

## Preprocessing dataset
dataset_train = pd.read_csv("datasets/google-stock-price/Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# for RNN, we need always reshape our data because RNN requires input shape as
# (batch_size, timesteps, input_dim)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

## Initialising the RNN
regressor = Sequential()

# Add the first LSTM layer and Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Add the second LSTM layer and Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Add the third LSTM layer and Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Add the fourth LSTM layer and Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compile the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the training Set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

## Making the predictions and visualising the results
dataset_test = pd.read_csv("datasets/google-stock-price/Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)

# Feature Scaling
inputs = sc.transform(inputs)

# Creating a data structure with 60 timesteps and 1 output
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
# for RNN, we need always reshape our data because RNN requires input shape as
# (batch_size, timesteps, input_dim)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print("Predicted stock price: {}".format(predicted_stock_price))

# Let us visualize the plots
plt.plot(real_stock_price, color='red', label="Real Google Stock Price")
plt.plot(predicted_stock_price, color='blue', label="Predicted Google Stock Price")
plt.title('Google Stock Price')
plt.xlabel('Time')
plt.ylabel('Open Stock Price')
plt.show(block=True)

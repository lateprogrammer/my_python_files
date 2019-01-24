# Recurrent Neural Network

# Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2] # Selecting only the column with index 1

# Feature Scaling

# There are two types of feature scaling: standardisation and normalisation
# Standardisation - Substract all your observations by mean value and divide by S.D
# Normalisation - Substract all your observations by min value & divide by (max-min)
# For the stock prices, normalisation is recommended and more relevant
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) # All the stock prices will be between 0 & 1
training_set_scaled = sc.fit_transform(training_set) # Apply fit and transform

# Creating a data structure with 60 timesteps and 1 output

# Why 60 timesteps - at any time t, the stock prices will look at 60 values before t
# Based on the 60 timesteps of the past information, it will try to predict t+1
X_train = []
y_train = []
for i in range (60, 1258):
    # The upper bound is excluded with first 60 stocks 
    X_train.append(training_set_scaled[i-60:i,0])
    # Rest of the stocks
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
    
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing Keras package from Tensorfow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising RNN
regressor = Sequential() # Regressor, Since we are predicting continuous output

# Adding the first LSTM layer and dropout regularisation

# 50 units represents the number of neurons to increase dimensionality
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) # 20% of the neurons are dropped out during training

# Adding the second LSTM layer and dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the thrid LSTM layer and dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer and dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = False)) # Since its final layer
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1)) # There is only one output which is the stock price

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training Set
regressor.fit(X_train, y_train, epochs = 200, batch_size = 32)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the Predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total [len(dataset_total)-len(dataset_test)-60:].values #.values to convert to numpy array
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range (60, 80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = "Real Google Stock Price")
plt.plot(predicted_stock_price, color = 'blue', label = "Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()































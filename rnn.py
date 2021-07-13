# Recurernt Neural Network with LSTM to predict the opening stock prices trends of Google based on historical data


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # Create a numpy array that contains one column which
# is the open stock price in the training data set (note: 2 is excluded but we had to put the range from
# 1 to 2 because we want the result to be a numpy array and not a vector despite the fact that it has only one column)

# Feature Scaling: 
# In case of RNN, it's recommended to do Feature Scaling using Normalization instead of Standardization (because of the use of Sigmoid functions)
from sklearn.preprocessing import MinMaxScaler
# All values after the Normalization will be between 0 and 1
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set) # Applies the Normalization to the trainng set

# Creating a data structure with 60 timesteps and 1 output
# At each timestep t, the network will have to remember (look back) at 60 timesteps before
# in order to learn some correlation to do the prediction
# The number 60 has been chosen based on experimenation: a larger number leads to overfitting: the network won't learn anything
# To do this, we will create two list. The first X_train will contain, for each timestep, the 60 previous stock prices
# That's why, we will start the loop at 60. For example: for timestep 60, we will record the stock prices from 0 to 59 (both included)
# the second list Y_train will contain the stock price at the designated timestep: in our example, the stock price at 
# the timestep 60 itself, we do that until the end of the dataset (we have 1257 data lines)
X_train = []
Y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0]) # i-60:i --> Remember the :i is excluded in the range
    Y_train.append(training_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train) # We have to convert the two lists into numpy arrays to be able to use them later on.

# Reshaping - to add indicator(s)
# So far, we have a two dimensional array of the size 1198x60: we have our input dataset as rows,
# and for each we have 60 timesteps. We need now to add indicators: which are the factors that we think
# are relevant for the model to look at when learning. For this example, we are going to use only one 
# indicator, which is the stock closing price (but we can use more if we want to ).
# We need to reshape the input to be a 3 dimensional array instead of 2 (to be able to add the indicator)
# That's why we are going to use the numpy reshape method
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN (Stacked LSTM)
# Importing the Keras libraries and packages
from keras.models import Sequential # Neural network object representing a sequence of layers
from keras.layers import Dense # Dense class to add the output layer
from keras.layers import LSTM # LSTM layer
from keras.layers import Dropout # Dropout regularization to avoid overfitting

# Initializing the RNN (using a sequence of layers - as opposed to computational graph)
regressor = Sequential() # The Recurrent Neural Network object is called regressor (not classifier) because this is a regression problem (we are predicting a continuous value (the stock price) and not a class)

# Adding the first LSTM layer and some Dropout regularization (to avoid overfitting)
# units: The number of neurons (memory units) in the first LSTM layer. We chose 50 (a large) number to have high-dimensionality
# return_sequences: Because we are going to have a stacked LSTM (we will add more LSTM layers), return sequences should be set to True
# for the last LSTM layer, we won't set the return_sequences because the default of this argument is False
# input_shape: the shape of the input. Note: we add only the shape of the timesteps and the indicators, because the observations
# don't need to be added (they are take into consideration by default)
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# To avoid overfitting we will instruct the neural network to dropout (ignore) a certain percentage of the neurons during the training
# a typical value for such rate is 20%, that's why we are initializing the Dropout class with 0.2
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularization (We don't need to specify the input shape of the previous layer: it's automatically recognized)
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regulariztion (We don't need to specify the input shape of the previous layer: it's automatically recognized)
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularization (We don't need to specify the input shape of the previous layer: it's automatically recognized)
# No return_sequences here because it should be False in the final LSTM layer which is the default value
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer
# The output layer is a fully connected to the previous layer
# units: The number of dimensions (neurons) that need to be in the output layer
# Since we are predicting a real value (the stock price), it's only one dimension (one neuron)
regressor.add(Dense(units=1))

# Part 3 - Making the predictions and visualizing the results

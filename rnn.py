# Recurernt Neural Network with LSTM to predict the opening stock prices *trends* (not the actual stock prices) of Google based on historical data


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set (The google stock prices of the years: 2012 to 2016)
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

# Compiling the RNN
# A common optimizer to be used is the RMSProp optimizer. It's a advanced stochastic gradient descent optimizer, according to the Keras documentation,
# is often a good choice for recurring neural networks
# But in this problem, we are not using the RMSProp optimizer. Experimenting lead to the use the Adam Optimizer, it's alwasy a safe choice that lead to better results.
# loss: since we are dealing with the regression problem, the loss will be the mean squared error (between the prediction and the real observation)
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training Set
# epochs: 100 is a number that was experimented with and lead to good results (suitable to 5 years of training data of the google stock price)
# batch_size: 32 a magic number that lead to good results
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and visualizing the results
# Getting the real stock price of 2017 (from the file Google_Stock_Price_Test.csv)
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of January 2017
# Remember that to predict the stock price of a day in January 2017, we will need the stock price of the previous 60 financial days
# And to have these days, we will need both the training set and the testing set, because some of the days will be in the testing set, 
# but some in the training set (because they are in the past!). Therefore, we will need some sort of concatenation. But
# remember: the test set should never be scaled (they should be as they are). So we are going to concatenate the original 
# data frames (read from the CSV files) of the training set and the testing set, then apply scaling to that and use it

# Will contain both the training set and the testing set
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0) # concatenation along the x axis of the open stock prices of both the training set and the testing set
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 :].values # Dataframe (not shaped as a 2D numpy array) of the inputs to the regressor which will be from 60 places before the beginning of the testing set plus the whole testing set
# it must have the shape of observations in rows and one column
inputs = inputs.reshape(-1, 1) # Takes the input data as rows, and creates a column a new dimension. Hence coverting a 2D numpy array. We need that shape for the fit method later on
inputs = sc.transform(inputs) # Directly applies the transform method (without fit) because the sc (Scaling object) was already fitted to the training set above. We must apply the same scaling

# Restructuring the inputs (test data) so that they also have 60 timestep for each input (like what we have done with the training data. See above)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
# Since the regressor is trained to predict the scaled values of the stock price, we need to inverse the scaling of the predicted output
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price Jan 2017')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price Jan 2017')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
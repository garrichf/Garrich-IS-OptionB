import keras.layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
from sklearn.model_selection import train_test_split
from collections import deque # A type of queue which stands for double ended queue which allows the access to front and back of queue.
from parameters import *
# from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

DATA_SOURCE = "yahoo"
# Set scaler variable to MinMaxScalar
scaler = MinMaxScaler()

# Function for randomly shuffling the dataset
def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


# Load and Process Data
def loadData(company, trainStartDate, trainEndDate, scale=True, splitByDate=False, shuffle=False, testSize=0.2, n_steps=numberOfSteps, lookup_step=1, feature_columns=columns):
    # Download data from yahoo finance
    # keepna=false parameter is so that if the rows have NaN it won't be downloaded
    data = yf.download(company, trainStartDate, trainEndDate, keepna=False)
    # Initialize dictionary to store the results
    result = {}
    # Add a copy of the current dataframe to the dictionary.
    result['data'] = data.copy()

    # confirms that each column specified in feature_columns exist inside the dataframe
    for col in feature_columns:
        assert col in data.columns, f"'{col}' does not exist in the dataframe."

    # Add date column if it does not exist, for some reason there is already a date column, but the format is weird.
    if "Date" not in data.columns:
        data["Date"] = data.index
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            data[column] = scaler.fit_transform(np.expand_dims(data[column].values, axis=1))
            column_scaler[column] = scaler

    # scaling is done to convert the data stored in the dataframe to be within a specified range so
    # that the model can compare and learn
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

        # adds a future column in the dataframe, the .shift(-lookup_step)
        # shifts the values of the columns up. specified by -lookup_step
        data['Future'] = data['Adj Close'].shift(-lookup_step)

        # copies the last lookup step rows and saves it in last_sequence.
        last_sequence = np.array(data[feature_columns].tail(lookup_step))

        # drops rows which has NaN in their row.
        data.dropna(inplace=True)

        # initialize sequence_data list
        sequence_data = []
        # initialize sequences double ended list which has maximum length of n_steps
        # if maximum length is reached, the oldest data will be overwritten by new data
        sequences = deque(maxlen=n_steps)
        # entry is the combination of feature_columnns and the dates
        # target is the future values.
        for entry, target in zip(data[feature_columns + ["Date"]].values, data['Future'].values):
            sequences.append(entry)
            # if length of sequences has reached n_steps it will append array of sequences with target to sequence_data.
            # this is so that the sequence_data can show the data in which, X amount of sequences will result in the
            # target value to be achieved. To be used to train the model later on. In the current case it will use
            # the past 50 days of data to determine the future value.
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])

        # list object
        # combines the feature columns sequences with last_sequence and creates a list
        last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
        # the list is then converted to a numpy array with the type float32
        last_sequence = np.array(last_sequence).astype(np.float32)
        # save the array into the result dictionary as 'last_sequence'
        result['last_sequence'] = last_sequence

        # initialize list for X and y
        X, y = [], []
        # appends sequence to list X and target to list y
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)
        # converts the list to numpy arrays
        X = np.array(X)
        y = np.array(y)
        # Split by Date
        if splitByDate:
            # split the dataset into training & testing sets by date (not randomly splitting)
            train_samples = int((1 - testSize) * len(X))
            # the colons here mean [: from beginning of dataframe. :] till the end of dataframe
            result["X_train"] = X[:train_samples]
            result["y_train"] = y[:train_samples]
            result["X_test"] = X[train_samples:]
            result["y_test"] = y[train_samples:]
            if shuffle:
                # shuffle the datasets for training (if shuffle parameter is set)
                # this function is defined before the load_data
                shuffle_in_unison(result["X_train"], result["y_train"])
                shuffle_in_unison(result["X_test"], result["y_test"])
        # if split by date is false then it will randomly split the data following specified test Size
        else:
            # split the dataset randomly between X_train, X_test, y_train, y_test using train_test_split
            result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=testSize, shuffle=shuffle)

        # get the list of test set dates
        dates = result["X_test"][:, -1, -1]
        # retrieve test features from the original dataframe
        result["test_df"] = result["data"].loc[dates]
        # remove duplicated dates in the testing dataframe
        result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
        # remove dates from the training/testing sets & convert to float32
        result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    return result

data = loadData(company, trainStart, trainEnd, columns)
# Convert them into an array
x_train, y_train = data['X_train'], data['y_train']
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# We now reshape x_train into a 3D array(p, q, 1); Note that x_train 
# is an array of p inputs with each input being a 2D array 

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

# model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True, batch_input_shape=(None, numberOfSteps, len(columns))))
# This is our first hidden layer which also spcifies an input layer.
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(keras.layers.LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
# print(type(x_train))
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
TEST_START = '2020-01-02'
TEST_END = '2022-12-31'

test_data = yf.download(company, start=trainStart, end=trainEnd, progress=False)

# The above bug is the reason for the following line of code
test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line
# Reshapes 1D array to a 2D array.

model_inputs = scaler.transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period 
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------


real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??
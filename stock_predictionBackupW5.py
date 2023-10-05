import keras.layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
from sklearn.model_selection import train_test_split
from collections import \
    deque  # A type of queue which stands for double ended queue which allows the access to front and back of queue.
from parameters import *
# from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer
import mplfinance as mpf
import os
import numpy as np



# turns off error message
pd.options.mode.chained_assignment = None

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
def loadData(company, trainStartDate, trainEndDate, scale=True, splitByDate=False, shuffle=False, testSize=0.3,
             n_past=pastDays, lookup_step=1, feature_columns=columns):
    # Download data from yahoo finance
    # keepna=false parameter is so that if the rows have NaN it won't be downloaded
    data = yf.download(company, trainStartDate, trainEndDate, keepna=False)
    # Initialize dictionary to store the results
    result = {}
    # Add a copy of the current dataframe to the dictionary.
    result['data'] = data.copy()

    # if saveData:
    #     data.to_csv(company_data_filename)

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
        sequences = deque(maxlen=n_past)
        # entry is the combination of feature_columnns and the dates
        # target is the future values.
        for entry, target in zip(data[feature_columns + ["Date"]].values, data['Future'].values):
            sequences.append(entry)
            # if length of sequences has reached n_steps it will append array of sequences with target to sequence_data.
            # this is so that the sequence_data can show the data in which, X amount of sequences will result in the
            # target value to be achieved. To be used to train the model later on. In the current case it will use
            # the past 50 days of data to determine the future value.
            if len(sequences) == n_past:
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
        # print(y)
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
        # remove dates from the training/testing sets & con3vert to float32
        result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

        # print(result)
        print(result['X_train'])
    return result


def buildModel(x_train_data, y_train_data, n_past, no_features, units=256, cell=LSTM, no_layers=2, dropout=0.3, loss='mean_absolute_error',
               optimizer="rmsprop", no_epochs=25, size_batch=32):
    # Use the sequential model type
    model = Sequential()
    print(cell)
    # a for loop to add layers based on the specified number of layers
    for i in range(no_layers):
        if i == 0:
            # The first layer, add a layer with return_sequences=True
            # batch_input_shape helps determine the input shape as in the number of inputs
            # the format of batch_input_shape is supposed to be
            # batch_input_shape(batch_size, number of steps and number of features)
            model.add(cell(units, return_sequences=True, batch_input_shape=(None, n_past, no_features)))
        elif i == no_layers - 1:
            # The last layer, add a recurrent layer with return_sequences=False. This if false because we want
            # one single final output instead of multiple, return_sequences can be true depending on the use
            # case.
            model.add(cell(units, return_sequences=False))
        else:
            # The layers in between first and last, add a layer with return_sequences=True
            model.add(cell(units, return_sequences=True))

        # a dropout layer is introduced, twhe function of this is to prevent overfitting
        # if a dropout rate of 0.2 that means that 20% of the data in this layer will be
        # dropped out or set to 0.
        model.add(Dropout(dropout))

    # Add a Dense layer with one output unit and linear activation for regression,
    # linear activation, produces an output that is directly proportional to the input.
    model.add(Dense(1, activation="linear"))

    # Compile the model with specified loss, metrics, and optimizer
    model.compile(loss=loss, metrics=['mean_absolute_error'], optimizer=optimizer)

    model.fit(x_train_data, y_train_data, epochs=no_epochs, batch_size=size_batch)


    # Return the constructed model
    return model


data = loadData(company, trainStart, trainEnd, columns)


def predict_future(last_sequence, n_days):
    x_sequence = last_sequence  # the last sequence in the dataframe
    predicted_prices = []  # a list to keep the predicted prices

    # use a for loop to iterate through the number of days into the future
    for i in range(n_days):
        # Reshape the input sequence for the model into a 3D model which is suitable for prediction.
        # the first digit 1 refers to that there is one day of data at a single time
        # the -1 is basically the time steps
        # len(columns) is basically the number of attributes
        x_sequence = x_sequence.reshape(1, -1, len(columns))
        # Make a prediction of the next day
        next_day_price = model.predict(x_sequence)

        # Store the prediction into the list of predicted prices
        predicted_prices.append(next_day_price)

        # Update the input sequence for the next iteration (shift and append the prediction)
        # the data on one end will be removed and the new data is added.
        # the -1 in shift means shift the valeus to the left
        x_sequence = np.roll(x_sequence, shift=-1, axis=1)
        # this line is for adding the new price predicted to the end of the list
        x_sequence[0, -1, -1] = next_day_price

    # Inverse transform the predicted prices to get them in their original scale
    # Because the original dataframe when scaling was a 2D array with 5 columns, the data
    # has to be shaped accordingly to prevent errors and for inverse scaling to work.
    # after the data is back to its normal values, i reshaped the data back into a 1D array.
    predicted_prices = np.array(predicted_prices).reshape(-1, len(columns))
    predicted_prices = data['column_scaler']['Adj Close'].inverse_transform(predicted_prices)
    predicted_prices = np.array(predicted_prices).reshape(-1)

    # Print out the predicted closing prices for X consecutive days into the future
    # formatted to the 2 decimal points
    for i in range(n_days):
        print(f"Day {i + 1}: Predicted Closing Price: AUD${predicted_prices[i]:.2f}")

    print("Lowest Price")
    print(min(predicted_prices))
    print("Average Price")
    print(sum(predicted_prices)/len(predicted_prices))
    print("Highest Price")
    print(max(predicted_prices))

if predict:
    # Convert them into an array
    x_train, y_train = data['X_train'], data['y_train']

    model = buildModel(x_train, y_train, pastDays, len(columns), units=unitSize, cell=DLType, no_layers=numberOfLayers,
                       dropout=DORate, loss=Loss, optimizer=Optimizer, no_epochs=Epochs, size_batch=BatchSize)
    # ------------------------------------------------------------------------------

    # Make predictions on test data
    y_test_pred = model.predict(data['X_test'])
    # Inverse transform the scaled data to get the actual prices
    # y_test_actual = data['column_scaler']['Adj Close'].inverse_transform(data['y_test'].reshape(-1, 1))
    # y_test_pred_actual = data['column_scaler']['Adj Close'].inverse_transform(y_test_pred)

    # y_test_actual = scaler.fit_transform(np.expand_dims(data['test_df']['Adj Close'].values, axis=1))
    y_test_actual = data['y_test']

    # Plot the test predictions
    plt.figure(figsize=(10, 6))
    plt.plot(data['test_df'].index, y_test_actual, label='Actual Price')
    plt.plot(data['test_df'].index, y_test_pred, label='Predicted Price')
    plt.title(f'{company} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    predict_future(data["last_sequence"], futureDays)

def candleGraph(dataframe, noOfTradingDays, scaleCandle):
    # get the columns High, Low, Open and Close from the dataframe, this is because
    # a candle graph needs this four parameters
    candleData = dataframe[['High', 'Low', 'Open', 'Close']]
    # index the data by the date.
    candleData.index.name = 'Date'
    # Set the title for the graph/plot
    candleTitle = f'{company},Stock Market Data from {trainStart} - {trainEnd}'

    # if number of trading days specified is greater than or equal to 1 then use the number
    # of days specified, else, it will just give the data of the entire dataframe.
    if noOfTradingDays >= 1:
        # get data only from no of past consecutive trading days from the current day
        candleData = candleData[-noOfTradingDays:]
        # update the title to this if using trading days specification
        candleTitle = f'{company}, CandleStiock Plot of Stock Market Data for Last {noOfTradingDays} Trading Days'
        # if scaling is wanted, then use this
    if scaleCandle:
        for col in candleData:
            candleData[col] = scaler.fit_transform(np.expand_dims(candleData[col].values, axis=1))
    # plot the data using candle type and binance style to see the red and greens.
    # title is set to candleTitle variable which was set above, and the label on the y
    # axis is "Price($)"
    mpf.plot(candleData, type='candle', style='binance', title=candleTitle, ylabel='Price ($)')
    plt.show()


def boxPlot(dataframe, noOfTradingDays):
    # gets the specified columns for each box plot needed
    boxData = dataframe[['High', 'Low', 'Open', 'Close', 'Adj Close']]
    # indexed by the date
    boxData.index.name = 'Date'

    # below follows the same format as the candlestick graph
    boxTitle = f'{company},Box Plot of Stock Market Data from {trainStart} - {trainEnd}'
    if noOfTradingDays >= 1:
        boxData = boxData[-noOfTradingDays:]
        boxTitle = f'{company}, Box Plot of Stock Market Data for Last {noOfTradingDays} Trading Days'
    # Create a box plot
    plt.boxplot(boxData)

    # Set labels and title
    plt.xlabel('Data')
    plt.ylabel('Price')
    plt.title(boxTitle)

    # this is to set the labels for the boxplots inside the graph.
    plt.xticks(range(1, len(boxData.columns) + 1), boxData.columns)

    # Show the plot
    plt.show()


# candleGraph(data['data'], num_trading_days, scaleCandle)
# boxPlot(data['data'], num_trading_days)


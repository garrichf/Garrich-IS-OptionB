import time
import os
import tensorflow as tf
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU


# ===========================================================================
# Loading and Processing Data
# Specifies company name
company = "TSLA"

# Training start and End Dates
# start = '2012-01-01', end='2017-01-01'
trainStart = '2015-01-01'
trainEnd = '2022-01-01'

# date now
date_now = time.strftime("%Y-%m-%d")

columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

# saving dataframe  
saveData = True
company_data_filename = os.path.join("data", f"{company}_{date_now}.csv")

scale = True
splitByDate = True

# Shuffling the dataframe (False by default)
Shuffle = False
# ===========================================================================

# ===========================================================================

# Building The Model
unitSize = 256
numberOfLayers = 4
Optimizer = 'adam'
Loss = 'mean_squared_error'
# Deep Learning Type
DLType = GRU
# Dropout Rate
DORate = 0.2
pastDays = 50
futureDays = 50

# ===========================================================================

# ===========================================================================

# do you want to predict?
predict = True
Epochs = 5
BatchSize = 32
futureprediction = True

# ===========================================================================


# ===========================================================================

# Plotting Graphs
scaleCandle = False
# Specify the number of trading days you want (0 = Off)
num_trading_days = 0

# ===========================================================================

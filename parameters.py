import time
import tensorflow as tf

# Specifies company name
company = "TSLA"

# Training start and End Dates
# start = '2012-01-01', end='2017-01-01'
trainStart = '2015-01-01'
trainEnd = '2020-01-01'

columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

#
numberOfSteps = 50

# Shuffling the dataframe (False by default)
Shuffle = False
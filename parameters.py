import time
import os
import tensorflow as tf

# do you want to predict?
predict = False

# Specifies company name
company = "TSLA"

# Training start and End Dates
# start = '2012-01-01', end='2017-01-01'
trainStart = '2015-01-01'
trainEnd = '2022-01-01'

# date now
date_now = time.strftime("%Y-%m-%d")

columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# saving dataframe
saveData = True
company_data_filename = os.path.join("data", f"{company}_{date_now}.csv")

scale = True;
splitByDate = True;

numberOfSteps = 100

scaleCandle = False;
# Specify the number of trading days you want (0 = Off)
num_trading_days = 0

# Shuffling the dataframe (False by default)
Shuffle = False
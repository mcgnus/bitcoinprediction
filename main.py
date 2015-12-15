from utilities import *
import random
# import matplotlib.pyplot as plt
import operator
from sklearn import linear_model
import numpy as np
import time

# --- import the trade data from file
print "import data"

# filename = "bitstamp_small.csv"
filename = "bitstampUSD.csv"
# filename = "btc100000.csv"
tstart = time.clock()
trades = readtrades(filename)
tstop = time.clock()
print str(tstop - tstart)
# --- calculate indicators
print "calculate indicators"
tstart = time.clock()
prices = trades["price"]
volume = trades["volume"]

mavg = moving_average(prices, 300)
emaslow, emafast, emadiff = moving_average_convergence(prices, 2600, 1200)
rsi = relative_strength(prices, 140)
tstop = time.clock()
print str(tstop - tstart)
# --- generate dataset
print "generate dataset"
tstart = time.clock()
future_steps = 1000
indics = []
future_prices = []
for idx in range(len(prices)-future_steps):
    indics.append([prices[idx],
                  volume[idx],
                  mavg[idx],
                  emaslow[idx],
                  emafast[idx],
                  emadiff[idx],
                  rsi[idx]])
    future_prices.append(prices[idx + future_steps])
    tstop = time.clock()
print str(tstop - tstart)
# --- mix dataset and split for training and testing
print "mix and split dataset"
tstart = time.clock()
training_fraction = 0.8

dataset = zip(indics, future_prices)
training_index = int(len(dataset)*training_fraction)

random.shuffle(dataset)

training_data_input, training_data_output = zip(*dataset[:training_index])
test_data_input, test_data_output = zip(*dataset[training_index:])
tstop = time.clock()
print str(tstop - tstart)

# --- train model
print "training model on dataset"
tstart = time.clock()
model = linear_model.LinearRegression()
model.fit(training_data_input, training_data_output)
tstop = time.clock()
print str(tstop - tstart)

# --- predict prices from model
print "predicting values"
tstart = time.clock()
pred = model.predict(test_data_input)
tstop = time.clock()
print str(tstop - tstart)

# --- calculate error and compare to acutal output
print "calculating error"
tstart = time.clock()
errs = map(operator.sub, test_data_output, pred)
errs = map(abs, errs)
errs = map(operator.div, errs, pred)
errs = map(lambda el: el*100, errs)
tstop = time.clock()
print str(tstop - tstart)


print "mean error", str(np.mean(errs)), "%"


# for x in zip(pred, test_data_output, errs):
#    print x

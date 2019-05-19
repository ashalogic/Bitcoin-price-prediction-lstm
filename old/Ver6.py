# import json
# import requests
import numpy as np
import pandas as pd
import keras.models as md
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import LSTM
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Embedding
# import matplotlib.pyplot as plt

series = pd.read_csv("test_data.csv")
series = series.set_index("date")
print(series)
print("=========================================")

window_data = []
for idx in range(len(series) - 5):
    tmp = series[idx: (idx + 5)].copy()
    window_data.append(tmp.values)
# print(window_data)
print("=========================================")
print(np.array(window_data))

window_size = 5
series_s = series.copy()
for i in range(window_size):
    series = pd.concat([series, series_s.shift(-(i+1))], axis=1)
# print(series)
print("=========================================")
series.dropna(inplace=True)
series = series.iloc[:, :-1]
# print(series)
print("=========================================")
X_series = np.array(series)
print(X_series)
print("=========================================")
# Y_series = series_s[5:].values
# print(Y_series)

# print(series)
# print("=========================================")
# print(series)

# model = Sequential()
# model.add(LSTM(units=5, input_shape=(5, 1), return_sequences=True))
# model.add(Dropout(0.5))
# model.add(LSTM(units=256))
# model.add(Dropout(0.5))
# model.add(Dense(units=1))
# model.add(Activation('linear'))
# model.compile(loss='mae', optimizer='adam')
# # model.build()
# model.summary()

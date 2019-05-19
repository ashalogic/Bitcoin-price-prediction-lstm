import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import InputLayer
from keras.models import Sequential
from keras.layers import Activation


def read_from_json(path):
    jfile = open(path)  # Open File
    jstring = jfile.read()  # Read File
    jobject = json.loads(jstring)['Data']  # Convert jobject
    # Convert to DataFrame
    df = pd.DataFrame(jobject, columns=['time', 'close'])
    df = df.set_index('time')  # DataFrame Index to Time
    df.index = pd.to_datetime(df.index, unit='s')
    return df


def read_from_url():
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=2000')
    jobject = pd.DataFrame(json.loads(res.content)['Data'])
    # Convert to DataFrame
    df = pd.DataFrame(jobject, columns=['time', 'close'])
    df = df.set_index('time')  # DataFrame Index to Time
    df.index = pd.to_datetime(df.index, unit='s')
    return df


def convert_data_to_tup(data, window_size):
    temp_data = data.copy()
    for i in range(window_size):
        temp_data = pd.concat([temp_data, data.shift(-(i+1))], axis=1)
    temp_data.dropna(inplace=True)
    temp_data = temp_data.iloc[:, :-1]
    return temp_data


def hist_price_dl(coin_id=1, timeframe="5y", currency="USD"):
    '''It accepts coin_id, timeframe, and currency parameters to clean the historic coin data taken from COINRANKING.COM
    It returns a Pandas Series with daily mean values of the selected coin in which the date is set as the index'''
    r = requests.get("https://api.coinranking.com/v1/public/coin/" +
                     str(coin_id)+"/history/"+timeframe+"?base="+currency)
    # Reading in json and cleaning the irrelevant parts
    coin = json.loads(r.text)['data']['history']
    df = pd.DataFrame(coin)
    df['price'] = pd.to_numeric(df['price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    return df.groupby('timestamp').mean()['price']


def normalize_data(data):
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data.values.reshape(-1, 1))
    return data


def invers_normalize_data(data):
    temp_data = data.copy()
    temp_data = temp_data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    temp_data = min_max_scaler.fit_transform(temp_data)
    # X_Train = pd.DataFrame(x_tr_scaled)
    X_Train = np.array(X_Train)
    return


def build_model(X_Data):
    model = Sequential()
    model.add(LSTM(20, input_shape=(X_Data.shape[1], X_Data.shape[2])))
    model.add(Dropout(0.25))
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    model.compile(loss="mae", optimizer="adam")
    model.summary()
    return model


def normalise_zero_base(df):
    """ Normalise dataframe column-wise to reflect changes with
        respect to first entry.
    """
    return df / df.iloc[0] - 1


def extract_window_data(df, window=7, zero_base=True):
    """ Convert dataframe to overlapping sequences/windows of
        length `window`.
    """
    window_data = []
    for idx in range(len(df) - window):
        tmp = df[idx: (idx + window)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.close)
    return np.array(window_data)


# data = read_from_url()  # read data
data = hist_price_dl()

# split
sp = len(data)-10
Train = data[:sp]
Test = data[sp:]


w = 7


t = Test.copy()
xtest = extract_window_data(Test, w, True)
predss = t.close.values[:-w] * (xtest + 1)
predss = pd.Series(index=targets.index, data=preds)


# predss = t.close.values[:-w] * (preds + 1)
# predss = pd.Series(index=targets.index, data=preds)

X_Train = convert_data_to_tup(Train, w)
X_Test = convert_data_to_tup(Test, w)
Y_Train = Test[w:].values
Y_Test = Test[w:].values

X_Train_sc = normalize_data(X_Train)
X_Test_sc = normalize_data(X_Test)
Y_Train_sc = normalize_data(Y_Train)
Y_Test_sc = normalize_data(Y_Test)

X_Test.set_index()

print()

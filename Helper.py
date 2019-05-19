import json
import pandas
import numpy
import datetime
import requests
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation


class Funcs:

    def __init__(self):
        self.min_max_scaler = preprocessing.MinMaxScaler()

    # Last 5Y picton price only 'close' price
    def Get_Historicalprice(self):
        endpoint = 'https://min-api.cryptocompare.com/data/histoday'
        res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=2000')
        jobject = pandas.DataFrame(json.loads(res.content)['Data'])
        df = pandas.DataFrame(jobject, columns=['time', 'close'])
        df = df.set_index('time')
        df.index = pandas.to_datetime(df.index, unit='s')
        date_from = datetime.datetime.now() - datetime.timedelta(days=(5*365.24))
        date_from = date_from.strftime("%Y-%m-%d")
        # date_to = datetime.datetime.now()
        df = df.loc[date_from:]
        return df

    # Normalize a dataframe with min max scaler
    def Normalize(self, df):
        x = df.values
        x_scaled = self.min_max_scaler.fit_transform(x)
        df = pandas.DataFrame(x_scaled, df.index)
        return df

    # Denormalize a dataframe from last min max scaler
    def Denormalize(self, x_scaled):
        x_orginal = self.min_max_scaler.inverse_transform(x_scaled)
        return x_orginal

    # split Data to Train and Test
    def Split_Test_Train(self, df, Test_size=0.1):
        sp = len(df)-Test_size
        Train = df[:sp]
        Test = df[sp:]
        return Train, Test

    # Conver Time series to Supervised Learning
    def Convert_TS_To_SL(self, data, window_size):
        temp_data = data.copy()
        for i in range(window_size):
            temp_data = pandas.concat([temp_data, data.shift(-(i+1))], axis=1)
        temp_data.dropna(inplace=True)
        temp_data = temp_data.iloc[:, :-1]
        return temp_data

    def build_LSTM(self, window_size, features=1):
        model = Sequential()
        model.add(LSTM(20, input_shape=(window_size, features)))
        model.add(Dropout(0.25))
        model.add(Dense(units=1))
        model.add(Activation('linear'))
        model.compile(loss="mae", optimizer="adam")
        return model

    def build_NextDays(self, df, days, window_size, model, features=1):
        df = df[len(df)-window_size:]
        predictdf = pandas.DataFrame()
        for i in range(0, days):
            dfarray = numpy.array(df.values)
            dfarray = dfarray.reshape(1, window_size, features)
            predictvalue = model.predict(dfarray)
            df = df.drop(df.index[0])
            last_date = df.iloc[[-1]].index
            last_date = last_date + datetime.timedelta(days=1)
            df = df.append(pandas.DataFrame(predictvalue, index=last_date))
            predictdf = predictdf.append(
                pandas.DataFrame(predictvalue, index=last_date))
        return predictdf

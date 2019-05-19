import json
import pandas
import datetime
import requests
from sklearn import preprocessing


class Helper:

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
    def Denormalize(self, df):
        x = df.values  # returns a numpy array
        x_orginal = self.min_max_scaler.inverse_transform(x)
        df = pandas.DataFrame(x_orginal, df.index)
        return df

    def Split_Test_Train(self, df, Test_size=0.1):
        # split
        sp = len(df)-Test_size
        Train = df[:sp]
        Test = df[sp:]
        return Train, Test

import json
import pandas
import datetime
import requests


class Helper:
    def Get_Historicalprice():
        endpoint = 'https://min-api.cryptocompare.com/data/histoday'
        res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=2000')
        jobject = pandas.DataFrame(json.loads(res.content)['Data'])
        df = pandas.DataFrame(jobject, columns=['time', 'close'])
        df = df.set_index('time')
        df.index = pandas.to_datetime(df.index, unit='s')
        date_from = datetime.datetime.now() - datetime.timedelta(days=(5*365.24))
        date_to = datetime.datetime.now()
        print(date_from.strftime("%Y-%m-%d"))
        f = now-(days=365)
        df = df.loc['2014-5-17':'2019-5-17]
        return df

    def Get_x():
        return

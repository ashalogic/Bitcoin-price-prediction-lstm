import json
import requests
import numpy as np
import pandas as pd
import keras.models as md
import matplotlib.pyplot as plt

#json_file = open("Data_2000.json")
#json_string = json_file.read()
#hist = pd.DataFrame(json.loads(json_string)['Data'])
#hist = hist.set_index('time')
#hist.index = pd.to_datetime(hist.index, unit='s')
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=2000')
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')

def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data
def line_plot(line1, line2, label1=None, label2=None, title=''):
    #fig, ax = plt.subplots(1, figsize=(16, 9))
    plt.plot(line1, label=label1, linewidth=2)
    plt.plot(line2, label=label2, linewidth=2)
    plt.ylabel('price [USD]', fontsize=14)
    plt.title(title, fontsize=18)  

train, test = train_test_split(hist, test_size=0.1)
line_plot(train.close, test.close, 'training', 'test', 'BTC')

#plt.plot(train.close, label="Train", linewidth=2)
#plt.plot(test.close, label="Test", linewidth=2)
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
        window_data.append(tmp.values)
    return np.array(window_data)
def prepare_data(df, window=7, zero_base=True, test_size=0.1):
    """ Prepare data for LSTM. """
    # train test split
    train_data, test_data = train_test_split(df, test_size)
    
    # extract window data
    X_train = extract_window_data(train_data, window, zero_base)
    X_test = extract_window_data(test_data, window, zero_base)
    
    # extract targets
    y_train = train_data.close[window:].values
    y_test = test_data.close[window:].values
    if zero_base:
        y_train = y_train / train_data.close[:-window].values - 1
        y_test = y_test / test_data.close[:-window].values - 1
    return train_data, test_data, X_train, X_test, y_train, y_test

train, test, X_train, X_test, y_train, y_test = prepare_data(hist)

#load json and create model
json_file = open('BTCModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = md.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("BTCModel.h5")
print("Loaded model from disk")


# convert change predictions back to actual price
targets = test["close"][7:]
preds = loaded_model.predict(X_test).squeeze()
# convert change predictions back to actual price
preds = test.close.values[:-7] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
n = 30
#line_plot(targets[-n:], preds[-n:], 'actual', 'prediction')
plt.plot(preds[-n:], label="prediction", linewidth=1)

def predddd(mmmm,p):
    # convert change predictions back to actual price
    targets = p["close"][7:]
    preds = loaded_model.predict(mmmm).squeeze()
    # convert change predictions back to actual price
    preds = p.close.values[:-7] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    n = 30
    #line_plot(targets[-n:], preds[-n:], 'actual', 'prediction')                                v
    plt.plot(preds[-n:], label="prediction", linewidth=1)
    
d1 = predddd(extract_window_data(preds),preds)
d2 = predddd(extract_window_data(d1),d1)
d3 = predddd(extract_window_data(d2),d2)
d4 = predddd(extract_window_data(d3),d3)
d5 = predddd(extract_window_data(d4),d4)
d6 = predddd(extract_window_data(d5),d5)
d7 = predddd(extract_window_data(d6),d6)

plt.legend(loc = 'best', fontsize = 18)
plt.show()

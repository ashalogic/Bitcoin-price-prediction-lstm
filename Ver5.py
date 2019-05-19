import json
import requests
import numpy as np
import pandas as pd
import keras.models as md
import matplotlib.pyplot as plt

#Read File
json_file = open("Data_2000.json")
json_string = json_file.read()

#Load Data
hist = pd.DataFrame(json.loads(json_string)['Data'])


#Set Col time as index
hist = hist.set_index('time')


#change index from timestamp to Datetime
hist.index = pd.to_datetime(hist.index, unit='s')


def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data
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



#Create Data
window = 7
zero_base = False

train, test = train_test_split(hist, 0.1)




X_train = extract_window_data(train, window, zero_base)

print(train)
print(X_train)

X_test = extract_window_data(test, window, zero_base)

y_train = train.close[window:].values
y_test = test.close[window:].values
y_train = y_train / train.close[:-window].values - 1
y_test = y_test / test.close[:-window].values - 1

#print(train)
#print(test)
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)




#load json and create model
json_file = open('BTCModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = md.model_from_json(loaded_model_json)
loaded_model.load_weights("BTCModel.h5")
print("Loaded model from disk")

 #convert change predictions back to actual price
#targets = test["close"][window:]
#preds = loaded_model.predict(X_test).squeeze()
 #convert change predictions back to actual price
#preds = test.close.values[:-window] * (preds + 1)
#preds = pd.Series(index=targets.index, data=preds)
#n = 30

#line_plot(targets[-n:], preds[-n:], 'actual', 'prediction')
#plt.plot(preds[-n:], label="prediction", linewidth=1)

#plt.plot(y_train)
#plt.plot(y_test)
#plt.plot(preds)


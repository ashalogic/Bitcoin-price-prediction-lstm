import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import LSTM
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Embedding

# S.1 Read Data From API
# endpoint = 'https://min-api.cryptocompare.com/data/histoday'
# res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=2000')
# jobject = pd.DataFrame(json.loads(res.content)['Data'])

# S.2 Read Data From FILE
jfile = open('Data_2000.json')  # Open File
jstring = jfile.read()  # Read File
jobject = json.loads(jstring)['Data']  # Convert jobject
df = pd.DataFrame(jobject, columns=['time', 'close'])  # Convert to DataFrame
df = df.set_index('time')  # DataFrame Index to Time
df.index = pd.to_datetime(df.index, unit='s')  # Convert Timestamp To Date

# print(df.describe())  # DataFrame Information
# df.info()  # DataFrame Information
# print(df.head())  # Print Head
# print(df.tail())  # Print Tail

# Split Data To Train And Test
split_row = len(df) - int(0.1 * len(df))
Train = df[:split_row]
Test = df[split_row:]
print("=========================================")
print("TR Count Is : " + str(len(Train)))
print("=========================================")
print("TE Count Is : " + str(len(Test)))
print("=========================================")
# print(Train.tail())  # Print Head
# print(Test.head())  # Print Tail
# plt.plot(Train)
# plt.plot(Test)

# Data prepare
window_size = 5

# X_Train
X_Train = Train.copy()
for i in range(window_size):
    X_Train = pd.concat([X_Train, Train.shift(-(i+1))], axis=1)
X_Train.dropna(inplace=True)
X_Train = X_Train.iloc[:, :-1]

# X_Test
X_Test = Test.copy()
for i in range(window_size):
    X_Test = pd.concat([X_Test, Test.shift(-(i+1))], axis=1)
X_Test.dropna(inplace=True)
X_Test = X_Test.iloc[:, :-1]

# Normalize X_Test X_Train
x_tr = X_Train.values
x_te = X_Test.values
y_tr = Train[window_size:].values
y_te = Test[window_size:].values
min_max_scaler = preprocessing.MinMaxScaler()
x_tr_scaled = min_max_scaler.fit_transform(x_tr)
x_te_scaled = min_max_scaler.fit_transform(x_te)
y_tr_scaled = min_max_scaler.fit_transform(y_tr)
y_te_scaled = min_max_scaler.fit_transform(y_te)

X_Train = pd.DataFrame(x_tr_scaled)
X_Train = np.array(X_Train)

X_Test = pd.DataFrame(x_te_scaled)
X_Test = np.array(X_Test)

Y_Train = pd.DataFrame(y_tr_scaled)
Y_Train = np.array(Y_Train)

Y_Test = pd.DataFrame(y_te_scaled)
Y_Test = np.array(Y_Test)

# Reshape
X_Train = X_Train.reshape(len(X_Train), window_size, 1)
Y_Train = Y_Train.reshape(len(Y_Train), )

# LSTM
model = Sequential()
model.add(LSTM(20, input_shape=(X_Train.shape[1], X_Train.shape[2])))
model.add(Dropout(0.25))
model.add(Dense(units=1))
model.add(Activation('linear'))
model.compile(loss="mae", optimizer="adam")
model.summary()
print(X_Train.shape)
print(Y_Train.shape)

history = model.fit(X_Train, Y_Train, epochs=5, batch_size=4)

X_Test = X_Test.reshape(len(X_Test), window_size, 1)
predi = model.predict(X_Test)
print(predi)
print("_______________")
print(Y_Test)


predi = Test[window_size:].values * (predi + 1)
predi = pd.Series(index=Test[window_size:].index, data=predi)
plt.plot(Y_Test)
plt.plot(predi, label="prediction", linewidth=1)
plt.show()

print(len(Y_Test))
print(len(predi))
print(len(Test.tail()))
print(predi)
print(Test[window_size:].index.values)

import json
import numpy
import pandas
import requests
from matplotlib import pyplot
from datetime import timedelta
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation

# Read Data
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=2000')
jobject = pandas.DataFrame(json.loads(res.content)['Data'])
df = pandas.DataFrame(jobject, columns=['time', 'close'])
df = df.set_index('time')
df.index = pandas.to_datetime(df.index, unit='s')
df = df.loc['2014-5-17':'2019-5-17']
print(df)

# Normal
x = df.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pandas.DataFrame(x_scaled, df.index)
print(df)

# split
sp = len(df)-0
Train = df[:sp]
Test = df[sp:]

# window


def convert_data_to_tup(data, window_size):
    temp_data = data.copy()
    for i in range(window_size):
        temp_data = pandas.concat([temp_data, data.shift(-(i+1))], axis=1)
    temp_data.dropna(inplace=True)
    temp_data = temp_data.iloc[:, :-1]
    return temp_data


w = 7

X_Train = convert_data_to_tup(Train, w)
Y_Train = Train[w:].values
X_Test = convert_data_to_tup(Test, w)
Y_Test = Test[w:].values

print(X_Test)
print(Y_Test)

pyplot.plot(Train)
pyplot.plot(Test)
pyplot.show()

# Reshape
X_Train = numpy.array(X_Train.values)
X_Train = X_Train.reshape(len(X_Train), w, 1)
Y_Train = Y_Train.reshape(len(Y_Train), )

# Train
model = Sequential()
model.add(LSTM(20, input_shape=(X_Train.shape[1], X_Train.shape[2])))
model.add(Dropout(0.25))
model.add(Dense(units=1))
model.add(Activation('linear'))
model.compile(loss="mae", optimizer="adam")
model.summary()
history = model.fit(X_Train, Y_Train, epochs=5, batch_size=4)

# Reshape
X_Test = numpy.array(X_Test.values)
X_Test = X_Test.reshape(len(X_Test), w, 1)

# Test
p = model.predict(X_Test)
print(p)
print("\n_VS_\n")
print(Y_Test)

# plot

ry = pandas.DataFrame(Y_Test, index=Test[w:].index)
py = pandas.DataFrame(p, index=Test[w:].index)

pyplot.plot(ry)
pyplot.plot(py)
pyplot.show()

tailofdata = df.loc['2019-5-9':'2019-5-15']
tailofdataorg = tailofdata.copy()

for i in range(0, 10):
    sdd = numpy.array(tailofdata.values)
    sdd = sdd.reshape(1, 7, 1)
    ps = (model.predict(sdd))
    tailofdata = tailofdata.drop(tailofdata.index[0])
    last_date = tailofdata.iloc[[-1]].index
    last_date = last_date + timedelta(days=1)
    tailofdata = tailofdata.append(pandas.DataFrame(ps, index=last_date))
    tailofdataorg = tailofdataorg.append(
        pandas.DataFrame(ps[0], index=last_date))

print(tailofdata)
print(tailofdataorg)

df = df.append(tailofdataorg)

dddds = tailofdataorg.values  # returns a numpy array
dddd_scaleds = min_max_scaler.inverse_transform(dddds)
dddds = pandas.DataFrame(dddd_scaleds, tailofdataorg.index)
print(df)
dddd = min_max_scaler.inverse_transform()

pyplot.plot(df["2019-05-01":])
pyplot.plot(dddds)
pyplot.show()

model.e


def predict(day):

    # for 10 days pridict

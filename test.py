import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read Data
df = pd.read_csv('Bitcoin_USD_2014-5-11_2019-5-11_Close.csv')

# Split Data
train = df[0:1461].iloc[:,1:2].values
test = df[1461:].iloc[:,1:2].values

#print(train.head())
#print(train.tail())
#print(test.head())
#print(test.tail())

#Draw Data
#plt.plot(train["Price"],label='Train')
#plt.plot(test["Price"],label='Test')
#plt.title('Bitcoin Pice From 2014/05/10 To 2019/05/10')
#plt.xlabel('Days')
#plt.ylabel('1 Bitcoin To USD')
#plt.legend()
#plt.show()
dd = np.asarray(train.Price)

#Naive Forcast
y_hat = test.copy()
y_hat['naive'] = dd[len(dd) - 1]

#Average Forcast
y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train.Price.mean()

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
train_scaled = sc.fit_transform(train)
X_train = []
Y_train = []
for i in range(60,len(train_scaled)):
    X_train.append(train_scaled[i - 60:i,0])
    Y_train.append(train_scaled[i,0])
X_train,Y_train = np.array(X_train),np.array(Y_train)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
       
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)




#from sklearn.metrics import mean_squared_error
#from math import sqrt

#Naive_rms = sqrt(mean_squared_error(test.Price, y_hat.naive))
#Avg_rms = sqrt(mean_squared_error(test.Price, y_hat_avg.avg_forecast))
#print("Naive_rms : " + str(Naive_rms))
#print("Avg_rms : " + str(Avg_rms))

##plt.figure(figsize=(12,8))
#plt.plot(train.index,train['Price'],label="Train")

#plt.plot(test.index,test['Price'],label="Test")

#plt.plot(y_hat.index,y_hat['naive'],label="Naive Forcast")

#plt.plot(y_hat_avg.index,y_hat_avg['avg_forecast'],label="Avg Forecast")

#plt.legend()
#plt.show()







# ###########################################################
# Bitcoin Price Predicon
# ###########################################################
# Auther:   Ashalogic
# Version:  0.1
# Date      Created: 5/19/2019
# Brief:    Get last 5Y of bitcoin price ther draw and print
#           Historical data and pridict for 10 days
# ###########################################################

import numpy
import datetime
import pandas

from Helper import Funcs
from matplotlib import pyplot as plt

print(" ===========================================================")
print(" Welcome to Bitcoin Price Predicon ver 0.1")
print(" ===========================================================")

hp = Funcs()  # Init Hepler class
# ===========================================================
Data = hp.Get_Historicalprice()  # get data
print(" Data     Shape : " + str(Data.shape))
# ===========================================================
Org_data = Data.copy()  # Make copy of org data -_o
# ===========================================================
Data = hp.Normalize(Data)  # normalize data
# ===========================================================
LastdaysForTest = 180
Train, Test = hp.Split_Test_Train(Data, LastdaysForTest)  # splitdata
print(" Train    Shape : " + str(Train.shape))
print(" Test     Shape : " + str(Test.shape))
# ===========================================================
window_size = 7
features = 1
# ===========================================================
X_Train = hp.Convert_TS_To_SL(Train, window_size)
X_Test = hp.Convert_TS_To_SL(Test, window_size)
Y_Train = Train[window_size:].values
Y_Test = Test[window_size:].values
print(" X_Train  Shape : " + str(X_Train.shape))
print(" X_Test   Shape : " + str(X_Test.shape))
print(" Y_Train  Shape : " + str(Y_Train.shape))
print(" Y_Test   Shape : " + str(Y_Test.shape))
# ===========================================================
ep = 30
bs = 4
LSTM_Model = hp.build_LSTM(window_size, features)
LSTM_Model.summary()
# ===========================================================
X_Train = numpy.array(X_Train.values)
X_Train = X_Train.reshape(len(X_Train), window_size, features)
Y_Train = Y_Train.reshape(len(Y_Train), )
# ===========================================================
history = LSTM_Model.fit(X_Train, Y_Train, epochs=ep, batch_size=bs)  # Train
# ===========================================================
X_Test = numpy.array(X_Test.values)
X_Test = X_Test.reshape(len(X_Test), window_size, features)
Y_Test = Y_Test.reshape(len(Y_Test), )  # Reshape Y_Test for Evaluate
# ===========================================================
PY_Test = LSTM_Model.predict(X_Test)  # Predict Y_Test
# ===========================================================
history = LSTM_Model.evaluate(X_Test, Y_Test)  # Evaluate Model -_o
# ===========================================================
Y_Test = Y_Test.reshape(len(Y_Test), features)  # Reshape Y_Test back
# ===========================================================
print(Y_Test)
print(PY_Test)
plt.plot(hp.Denormalize(Y_Test), color='green', label='Test Real Price')
plt.plot(hp.Denormalize(PY_Test), color='orange', label='Test Predict Price')
plt.legend() 
plt.show()
# ===========================================================
daystopredict = 180
nextdays = hp.build_NextDays(Data, daystopredict, window_size, LSTM_Model)
print(hp.Denormalize(nextdays.values))
plt.plot(Data[365:])
plt.plot(nextdays)
plt.legend()
plt.show()
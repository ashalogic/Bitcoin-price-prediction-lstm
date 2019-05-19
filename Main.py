# ###########################################################
# Bitcoin Price Predicon
# ###########################################################
# Auther:   Ashalogic
# Version:  0.2
# Date      Created: 5/19/2019
# Brief:    Get last 5Y of bitcoin price ther draw and print
#           Historical data and pridict for 10 days
# ###########################################################

from Helper import Helper

print(" ===========================================================")
print(" Welcome to Bitcoin Price Predicon ver 0.2")
print(" ===========================================================")

hp = Helper()  # Init Hepler class
# ===========================================================
Data = hp.Get_Historicalprice()  # get data
print(" Data  Shape : " + str(Data.shape))
# ===========================================================
Org_data = Data.copy()  # Make copy of org data -_o
# ===========================================================
Data = hp.Normalize(Data)  # normalize data
# ===========================================================
Train, Test = hp.Split_Test_Train(Data, 0)  # splitdata
print(" Train Shape : " + str(Train.shape))
print(" Test  Shape : " + str(Test.shape))
# ===========================================================


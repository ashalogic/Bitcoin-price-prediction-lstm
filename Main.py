# ###########################################################
# Bitcoin Price Predicon
# ###########################################################
# Auther:   Ashalogic
# Version:  0.2
# Date      Created: 5/19/2019
# Brief:    Get last 5Y of bitcoin price ther draw and print
#           Historical data and pridict for 10 days
# ###########################################################

# from Helper import Helper as hp

import datetime


def Run():
    # Data = hp.Get_Historicalprice()
        # f = now-(days=365)
    date_from = datetime.datetime.now() - datetime.timedelta(days=(5*365.24))
    date_to = datetime.datetime.now()
    print(date_from.strftime("%Y-%m-%d"))

Run()

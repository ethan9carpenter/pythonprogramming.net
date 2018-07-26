import pandas as pd
from matplotlib import style, pyplot
import quandl
from tools import apiKey
from part7 import hpiBenchmark

style.use('fivethirtyeight')

def getThirtyYearMortgageData():
    data = quandl.get("FMAC/MORTG", trim_start='1975-01-01', authtoken=apiKey)
    data['Value'] = (data['Value'] - data['Value'][0]) / data['Value'][0] * 100.0
    #Resample daily, and then monthly so it will match the HPI data format
    data = data.resample('1D').mean() 
    data = data.resample('M').mean()
    
    return data

"""
states = pd.read_pickle('resources/fiftyStates2.pickle')
mortData = getThirtyYearMortgageData()
benchmark = hpiBenchmark()

mortData.columns = ['M30']
nationalData = benchmark.join(mortData)
stateData = states.join(mortData)

print(stateData.corr()['M30'].describe())
"""


import pandas as pd
from matplotlib import pyplot, style
from part7 import hpiBenchmark, apiKey, writeInitialStateData
from part13 import getThirtyYearMortgageData
import pickle
import quandl

style.use('fivethirtyeight')

def getSP500():
    data = quandl.get("MULTPL/SP500_REAL_PRICE_MONTH", trim_start='1975-01-01', authtoken=apiKey)
    data['Value'] = (data['Value'] - 
                    data['Value'][0]) / data['Value'][0] * 100.0
    #Resample daily, and then monthly so it will match the HPI data format
    data = data.resample('1D').mean()
    data = data.resample('M').mean()
    data.rename(columns={'Value': 'sp500'}, inplace=True)
    
    return data

def getGDP():
    data = quandl.get("BCB/4385", trim_start='1975-01-01', authtoken=apiKey)
    data['Value'] = (data['Value'] - data['Value'][0]) / data['Value'][0] * 100.0
    data = data.resample('M').mean()
    data.rename(columns={'Value': 'GDP'}, inplace=True)
    return data

def getUnemployment():
    data = quandl.get("USMISERY/INDEX", trim_start='1975-01-01', authtoken=apiKey)
    data['Unemployment Rate'] = (data['Unemployment Rate'] - 
                                 data['Unemployment Rate'][0]) / data['Unemployment Rate'][0] * 100.0
    #Resample daily, and then monthly so it will match the HPI data format
    data = data.resample('1D').mean() 
    data = data.resample('M').mean()
    data = data['Unemployment Rate']
    
    return data

#writeInitialStateData()
stateHPI = pd.read_pickle('resources/fiftyStates.pickle')
mortData = getThirtyYearMortgageData()
sp500data = getSP500()
unemployData = getUnemployment()
gdpData = getGDP()
benchmark = hpiBenchmark()

mortData.columns = ['M30']
HPI = benchmark.join([mortData, sp500data, unemployData, gdpData])
HPI = stateHPI.join(HPI)
HPI.dropna(inplace=True)

HPI.to_pickle('resources/HPI.pickle')






import pandas as pd
import pandas_datareader.data as web
import pickle
import json
import quandl

with open('Users/footballnerd12/API_keys.json') as file:
    apiKey = json.load(file)['quandl']

def getStateList():
    statesHMTL = pd.read_html('resources/states.html')[0][1][1:]
    abbreviations = []
    for abbreviation in statesHMTL:
        abbreviations.append(abbreviation)
    
    return abbreviations
        
def writeInitialStateData(rolling=True):
    mainData = pd.DataFrame()

    for abrv in getStateList():
        data = quandl.get("FMAC/HPI_"+abrv, authtoken=apiKey)
        data.columns = [abrv]
        
        if rolling:
            data[abrv] = (data[abrv] - data[abrv][0]) / data[abrv][0] * 100.0
        path = 'resources/fiftyStates.pickle'
        
        if mainData.empty:
            mainData = data
        else:
            mainData = mainData.join(data)
    
    
    with open(path, 'wb') as file:
        pickle.dump(mainData, file)

def hpiBenchmark():
    data = quandl.get("FMAC/HPI_USA", authtoken=apiKey)
    data.columns = ['US_HPI']
    
    data['US_HPI'] = (data['US_HPI'] - 
                             data['US_HPI'][0]) / data['US_HPI'][0] * 100.0
    return data

def getThirtyYearMortgageData():
    data = quandl.get("FMAC/MORTG", trim_start='1975-01-01', authtoken=apiKey)
    data['Value'] = (data['Value'] - data['Value'][0]) / data['Value'][0] * 100.0
    #Resample daily, and then monthly so it will match the HPI data format
    data = data.resample('1D').mean() 
    data = data.resample('M').mean()
    
    return data

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

def writeAllFactors():
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


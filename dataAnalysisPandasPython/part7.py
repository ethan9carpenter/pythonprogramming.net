import pandas as pd
import quandl, pickle

apiKey = 'rcYX2m1mjgcxsyx_skRb'

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

#writeInitialStateData(rolling=True)

import pandas as pd
import quandl, pickle

apiKey = 'rcYX2m1mjgcxsyx_skRb'

def getStateList():
    statesHMTL = pd.read_html('resources/states.html')[0][1][1:]
    quandlKeys = []
    for abbreviation in statesHMTL:
        quandlKeys.append('FMAC/HPI_'+abbreviation)
    
    return quandlKeys
        
def writeInitialStateData():
    mainData = pd.DataFrame()

    for key in getStateList():
        data = quandl.get(key, authtoken=apiKey)
        data.columns = [key]
        
        if mainData.empty:
            mainData = data
        else:
            mainData = mainData.join(data)

    with open('fiftyStates.pickle', 'wb') as file:
        pickle.dump(mainData, file)

with open('fiftyStates.pickle', 'rb') as file:
    data = pickle.load(file)
    data.to_pickle('pandaToPickle.pickle')
    data2 = pd.read_pickle('pandaToPickle.pickle')
    print(data, data2)


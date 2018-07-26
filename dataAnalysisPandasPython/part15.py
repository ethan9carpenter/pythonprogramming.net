import pandas as pd
from matplotlib import pyplot, style
import numpy as np
from statistics import mean

style.use('fivethirtyeight')

def createLabels(currentHPI, futureHPI):
    if futureHPI > currentHPI:
        return 1
    else:
        return 0

def movingAverage(data):
    return mean(data)


"""
housingData = pd.read_pickle('resources/HPI.pickle')
housingData = housingData.pct_change()

housingData.replace([np.inf, -np.inf], np.nan, inplace=True)
housingData['US_HPI_future'] = housingData['US_HPI'].shift(-1)
housingData.dropna(inplace=True)

housingData['label'] = list(map(createLabels, 
                                housingData['US_HPI'], housingData['US_HPI_future']))
housingData['movingAverageExample'] = housingData['M30'].rolling(10).apply(movingAverage, raw=False)

print(housingData)
"""



import pandas as pd
from collections import Counter
import numpy as np

def processForLabels(ticker, numDays=7):
    df = pd.read_csv('resources/mergedSP500.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(value=0, inplace=True)
    
    for i in range(1, numDays+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    df.fillna(value=0, inplace=True)
    
    return tickers, df

def buySellHold(*args, minChange=0.02):
    cols = list(args)
    
    for col in cols:
        if col > minChange:
            return 1
        elif col < minChange:
            return -1
    return 0

def getFeatureSets(ticker):
    tickers, df = processForLabels(ticker)
    
    df['{}_target'.format(ticker)] = list(map(buySellHold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]))
    
    values = df['{}_target'.format(ticker)].values.tolist()
    stringValues = [str(i) for i in values]
    print('Data Spread:', Counter(stringValues))
    
    df = _clean(df)
    dfValues = _getFormatted(df, tickers)
    
    x = dfValues.values
    y = df['{}_target'.format(ticker)].values
    
    return x, y, df

def _clean(df):
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    
    return df

def _getFormatted(df, tickers):
    dfValues = df[tickers].pct_change()
    dfValues = dfValues.replace([np.inf, -np.inf], 0)
    dfValues.fillna(0, inplace=True)
    
    return dfValues




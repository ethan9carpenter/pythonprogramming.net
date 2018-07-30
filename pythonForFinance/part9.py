import pandas as pd

def processForLabels(ticker, numDays=7):
    df = pd.read_csv('resources/mergedSP500.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(value=0, inplace=True)
    
    for i in range(0, len(1, numDays+1)):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    df.fillna(value=0, inplace=True)
    
    return tickers, df
        
    
    





















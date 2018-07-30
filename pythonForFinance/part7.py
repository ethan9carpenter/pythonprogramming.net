import pickle
import pandas as pd
from os.path import exists

def compileData():
    with open('resources/sp500tickers.pickle', 'rb') as file:
        tickers = pickle.load(file)
        
    mainDF = pd.DataFrame()
    
    for ticker in tickers:
        if exists('stockDFS/{}.csv'.format(ticker)):
            df = pd.read_csv('stockDFS/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
            df.rename(columns={'Close': ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)
            
            if mainDF.empty:
                mainDF = df
            else:
                mainDF = mainDF.join(df)
            
    mainDF.to_csv('resources/mergedSP500.csv')
    
compileData()
        
        














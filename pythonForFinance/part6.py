import bs4 as bs
import datetime as dt
import os
import pandas_datareader.data as web
import pickle
import requests
from part5 import saveSP500

def loadTickers(reload=False):
    if reload:
        tickers = saveSP500()
    else:
        with open('resources/sp500tickers.pickle', 'rb') as file:
            tickers = pickle.load(file)
    
    return tickers

def getYahooData(reload=False, forceUpdate=False):
    tickers = loadTickers(reload)
    
    if not os.path.exists('stockDFS'):
        os.makedirs('stockDFS')
    
    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()
    
    errors = []
    alreadySaved = []
    
    for ticker in tickers:
        #Save each time in case connection is broken
        if not os.path.exists('stockDFS/{}.csv'.format(ticker)) or forceUpdate:
            try:
                data = web.DataReader(ticker, 'morningstar', start, end)
                data.reset_index(inplace=True)
                data.set_index('Date', inplace=True)
                data.drop('Symbol', axis=1, inplace=True)
                data.to_csv('stockDFS/{}.csv'.format(ticker))
            except Exception:
                errors.append(ticker)
        else:
            alreadySaved.append(ticker)
    
    print('Already Saved:', alreadySaved)
    print('Errors:', errors)
            
#getYahooData()
        
    
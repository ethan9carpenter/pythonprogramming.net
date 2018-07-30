import pandas as pd
import pandas_datareader.data as web
import requests
import pickle
import bs4 as bs
import os
import datetime as dt

def saveSP500():
    req = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(req.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open('resources/sp500tickers.pickle', 'wb') as file:
        pickle.dump(tickers, file)
        
    return tickers

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

def compileData():
    with open('resources/sp500tickers.pickle', 'rb') as file:
        tickers = pickle.load(file)
        
    mainDF = pd.DataFrame()
    
    for ticker in tickers:
        if os.path.exists('stockDFS/{}.csv'.format(ticker)):
            df = pd.read_csv('stockDFS/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
            df.rename(columns={'Close': ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)
            
            if mainDF.empty:
                mainDF = df
            else:
                mainDF = mainDF.join(df)
            
    
    print(mainDF.head())
    mainDF.to_csv('resources/mergedSP500.csv')



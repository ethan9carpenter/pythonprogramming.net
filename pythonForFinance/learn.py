import pandas as pd
from collections import Counter
import numpy as np
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import warnings
import pickle
from statistics import mean
from re import split

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
        elif col < -minChange:
            return -1
    return 0

def getFeatureSets(ticker, printCount=False):
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
    
    if printCount:
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

def learn(ticker, testSize=.25, toprint=False):
    x, y = getFeatureSets(ticker)[0:1]
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=testSize)
    
    #===========================================================================
    #A VotingClassifier uses multiple classifiers to 'vote' on which label
    # should be selected. 
    #===========================================================================   
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    #clf = RandomForestClassifier()
    clf.fit(xTrain, yTrain)
    #===========================================================================
    # classifier.score() returns a % accuracy, not a correlation coefficient.
    # Remember that machine learning is used to identify a discrete value,
    # such as yes/no, buy/sell/hold, etc. rather than a regression so .90
    # would mean it correctly predicted 90% of the values.
    #===========================================================================
    conf = clf.score(xTest, yTest)
    predictions = clf.predict(xTest)
    
    if toprint:
        print('Accuracy:', conf)
        print('Predicted class counts:', Counter(predictions))
        print('\n')

    return conf, predictions

def learnAllAndSave(dateString):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        
        with open('resources/sp500tickers.pickle', 'rb') as file:
            tickers = pickle.load(file)
        
        accuracies = []
        path = 'results/modelResults_' + dateString + '.txt'
        
        for i, ticker in enumerate(tickers):
            if i%10 == 0:
                print(i)
            try:
                accuracy = learn(ticker, toprint=False)[0]
                accuracies.append(accuracy)
                
                with open(path, 'a') as file:
                    file.write('{}: {}\n'.format(ticker, accuracy))
                    print('{} accuracy: {}'.format(ticker, accuracy))
            except KeyError:
                print('Error with', ticker)
    
        with open(path, 'a') as file:
                    file.append('Average: {}'.format(mean(accuracies)))
                    print('\nAverage accuracy:', mean(accuracies))

def readSavedResults(path):
    with open(path, 'r') as file:
        lines = file.read()
    
    lines = split('\n', lines)[:-1]
    results = {}
    
    for line in lines:
        splitLine = split(': ', line)
        ticker = splitLine[0]
        number = float(splitLine[1])
        results[ticker] = number
    
    return results
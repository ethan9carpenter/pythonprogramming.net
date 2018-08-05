import quandl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import json
from math import ceil
import numpy as np
from matplotlib import style, pyplot
import datetime as dt
import time

style.use('fivethirtyeight')

with open('apiKeys.json', 'r') as file:
    keys = json.load(file)
    quandlKey = keys['quandl']

df = quandl.get('WIKI/GOOGL', authtoken=quandlKey)
df = df[['Adj. Open', 'Adj. Close', 'Adj. Low', 'Adj. High', 'Adj. Volume']]

df['hlVolatility'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low']
df['percentChange'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']

df = df[['Adj. Close', 'hlVolatility', 'percentChange', 'Adj. Volume']]

#print(df.head())

predictionCol = 'Adj. Close' #the column of what we want to predict
df.fillna(-99999, inplace=True)

daysOut = int(ceil(.01 * len(df)))

df['label'] = df[predictionCol].shift(-daysOut) #what is being predicted

featureDrops = ['label', 'Adj. Close']

X = np.array(df.drop(featureDrops, 1))
X = preprocessing.scale(X)#BE CAREFUL WHERE YOU PUT THIS
Xrecent = X[-daysOut:]  #test on most recent data
X = X[:-daysOut]  


df.dropna(inplace=True)
y = np.array(df['label'])

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=.25)

model = LinearRegression()#n_jobs is how many threads at once and increases speed, -1=max
#model = svm.SVR()

model.fit(xTrain, yTrain)
score = model.score(xTest, yTest)

predictionSet = model.predict(Xrecent)

print(predictionSet, score)

df['Forecast'] = np.nan
lastDate = df.iloc[-1:].index.date[0]
#print(type(lastDate), lastDate)
lastUnix = time.mktime(lastDate.timetuple())
secondsInDay = 86400.0
nextUnix = lastUnix + secondsInDay

for i in predictionSet:
    nextDate = dt.datetime.fromtimestamp(nextUnix)
    nextUnix += secondsInDay
    #clear columns and set last to the prediciton value
    df.loc[nextDate] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    
df['Adj. Close'].plot()
df['Forecast'].plot()
pyplot.legend(loc=4)
pyplot.xlabel('Date')
pyplot.ylabel('Price')
pyplot.show()

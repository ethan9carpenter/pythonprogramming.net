import quandl
import sklearn
import json
from math import ceil

with open('C:\\Users\\ethan\\OneDrive\\Desktop\\apiKeys.json', 'r') as file:
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

daysOut = int(ceil(.01 * len(df))) #predict out n # of days where n is ten % of the length of the df

df['label'] = df[predictionCol].shift(-daysOut)

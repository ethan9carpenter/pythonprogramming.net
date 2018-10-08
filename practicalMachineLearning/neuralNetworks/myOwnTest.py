from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('nba.csv')

toDrop = ['Tm', 'Lg', 'G', 'W/L%', 'L', 'Rk']
params = list(df.columns.drop(toDrop))
params.remove('W')
df.drop(toDrop, axis=1, inplace=True)

for row, season in enumerate(df['Season']):
    df['Season'][row] = str(season)[:4]

X = np.array(df.drop('W', axis=1))
y = np.array(df['W'])
X = preprocessing.scale(X)

xTrain, xTest, yTrain, yTest = train_test_split(X, y)


reg = MLPRegressor(max_iter=1000000)
#reg = LinearRegression()
reg.fit(xTrain, yTrain)

print(reg.score(xTest, yTest))
#pprint(dict(zip(params, reg.coef_)))
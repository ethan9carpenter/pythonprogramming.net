from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
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

reg = MLPRegressor(max_iter=10000)
reg = LinearRegression()
reg.fit(X, y)

print(reg.score(X, y))
pprint(dict(zip(params, reg.coef_)))
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import model_selection

housingData = pd.read_pickle('resources/HPI.pickle')
housingData = housingData.pct_change()

housingData.replace([np.inf, -np.inf], np.nan, inplace=True)
housingData['US_HPI_future'] = housingData['US_HPI'].shift(-1)
housingData.drop(columns=['US_HPI'], inplace=True)
housingData.dropna(inplace=True)

features = np.array(housingData.drop(columns=['US_HPI_future']))
labels = np.array(housingData['US_HPI_future'])

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(features, labels, test_size=.2)

regressor = LinearRegression()

model = regressor.fit(xTrain, yTrain)


print(model.coef_)
print(model.score(xTest, yTest))



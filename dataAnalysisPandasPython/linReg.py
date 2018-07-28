import pandas as pd
import numpy as np
from sklearn import svm, preprocessing, model_selection

housingData = pd.read_pickle('resources/HPI.pickle')
housingData = housingData.pct_change()

housingData.replace([np.inf, -np.inf], np.nan, inplace=True)

print(housingData)



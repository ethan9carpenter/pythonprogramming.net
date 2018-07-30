import pandas as pd
import numpy as np
from part15 import createLabels
from sklearn import svm, preprocessing, model_selection
from numpy import mean

housingData = pd.read_pickle('resources/HPI.pickle')
housingData = housingData.pct_change()

housingData.replace([np.inf, -np.inf], np.nan, inplace=True)
housingData['US_HPI_future'] = housingData['US_HPI'].shift(-1)
housingData.dropna(inplace=True)

housingData['label'] = list(map(createLabels, 
                                housingData['US_HPI'], housingData['US_HPI_future']))

def getSolution(testSize):
    features = np.array(housingData.drop(columns=['label', 'US_HPI_future']))
    features = preprocessing.scale(features)
    
    labels = np.array(housingData['label'])
    
    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(features, labels, test_size=testSize)
    
    classifier = svm.SVC(kernel='linear')
    classifier.fit(xTrain, yTrain)
    
    score = classifier.score(xTest, yTest)
    
    variables = list(housingData.columns)[0:-2]
    coefficients = list(classifier.coef_[0])
    weights = pd.DataFrame({'Variable': variables, 'Coefficient': coefficients})

    return {'score': score, 'weights': weights}

scores = []
for i in range(1000):
    solution = getSolution(.2)
    scores.append(solution['score'])
    #print(solution['score'])
    #print(solution['weights'])
print(mean(scores))



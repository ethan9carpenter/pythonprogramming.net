from sklearn import svm, preprocessing, model_selection
from numpy import mean
import numpy as np
import pandas as pd

def createLabels(currentHPI, futureHPI):
    if futureHPI > currentHPI:
        return 1
    else:
        return 0

def movingAverage(data):
    return mean(data)

def getAndFormatDataToLearn():
    housingData = pd.read_pickle('resources/HPI.pickle')
    housingData = housingData.pct_change()
    
    housingData.replace([np.inf, -np.inf], np.nan, inplace=True)
    housingData['US_HPI_future'] = housingData['US_HPI'].shift(-1)
    housingData.dropna(inplace=True)
    
    housingData['label'] = list(map(createLabels, 
                                    housingData['US_HPI'], housingData['US_HPI_future']))

def getSolution(testSize):
    housingData = getAndFormatDataToLearn()
    
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
import numpy as np
import pandas as pd
from collections import Counter
import warnings
import random

class KNearestNeighbors():
    def loadData(self, data):
        #Stores the data to be used for predictions
        self.data = data
            
    def predict(self, predictionInput, k=3):        
        if len(self.data) >= k:
                warnings.warn("k is set to a value less than total voting groups.")      
        distances = []
        
        #for each classification in the dictioray self.data
        for classification in self.data: 
            #for each list of features in the data for a given classification
            for features in self.data[classification]: 
                distance = np.linalg.norm(np.array(features) - np.array(predictionInput))
                distances.append([distance, classification])
                
        votes = []
        for i in sorted(distances)[:k]:
            votes.append(i[1])
            
        result = Counter(votes).most_common(1)
        result = result[0][0]
        
        return result

df = pd.read_csv('resources/breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop('id', axis=1, inplace=True)
fullData = df.astype(dtype=float).values.tolist()
random.shuffle(fullData)

testSize = .2
testLength = int(testSize * len(fullData))

trainSet = {2: [], 4: []}
testSet = {2: [], 4: []}
trainData = fullData[:-testLength]
testData = fullData[-testLength:]

for i in trainData:
    trainSet[i[-1]].append(i[:-1])
for i in testData:
    testSet[i[-1]].append(i[:-1])
    
numCorrect = 0
total = 0

model = KNearestNeighbors()
model.loadData(trainSet)
    
for classification in testSet:
    for featureSet in testSet[classification]:
        result = model.predict(featureSet, k=5)
        
        if result == classification:
            numCorrect += 1
        total += 1

print(numCorrect, total)





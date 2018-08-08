import numpy as np
from matplotlib import pyplot, style

style.use('fivethirtyeight')

class SupportVectorMachine:
    def __init__(self, visualtion=True):
        self.visualtion = visualtion
        self.colors = {1: 'r', -1: 'b'}
        
        if self.visualtion:
            self.figure = pyplot.figure()
            self.ax = self.figure.add_subplot(1, 1, 1)
        
    def fit(self, data, optimizationDegree=3):
        self.data = data
        # {||w||: [w, b]}
        optimizationDict = {}
        vectorTransforms = [[1, 1],
                            [1, -1],
                            [-1, 1],
                            [-1, -1]]
        
        self.setFeatureExtremes()
        stepSizes = [self.featureMax * (10 ** -i) for i in range(1, optimizationDegree+1)]
        #extremely expensive in runtime, doesn't need to be as precise as w
        bRangeMult, bMult = 5, 5
        
        lastOptimal = self.featureMax * 10 #w is initially [lastOPtimal, lastOPtimal]
        
        for step in stepSizes:
            w = np.array([lastOptimal, lastOptimal])
            optimized = False #allowed because it is a convex optimization
            
            while not optimized:
                pass
        
    def getFeautureExtremes(self):
        allFeatureValues = {}   
        for yi in self.data: #for each classification list in data
            for featureSet in self.data[yi]: #for each featureset in that list
                for feature in featureSet: #for each feaure in featureSet
                    allFeatureValues.append(feature)
                    
        self.featureMax = max(allFeatureValues)
        self.featureMin = min(allFeatureValues)
        del allFeatureValues
        
    def predict(self, features):
        # sign of x-dot-w + b
        dot = np.dot(np.array(features), self.w)
        label = np.sign(dot)
        
        return label

data = {-1: np.array([[1, 7],
                      [2, 8],
                      [3, 8]]),
        1: np.array([[5, 1],
                     [6, -1],
                     [7, 3]])}

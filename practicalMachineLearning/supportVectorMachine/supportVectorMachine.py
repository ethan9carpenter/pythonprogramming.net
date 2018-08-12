import numpy as np
from matplotlib import pyplot, style

style.use('fivethirtyeight')

class SupportVectorMachine:
    def __init__(self, visualtion=True):
        self.visualtion = visualtion
        self.colors = {1: 'r', -1: 'b', 0: 'g'}
        
        if self.visualtion:
            self.figure = pyplot.figure()
            self.ax = self.figure.add_subplot(1, 1, 1)
    
    def setFitSteps(self, optimizationDegree):
        #precision of w
        stepSizes = [self.featureMax * (10 ** -i) for i in range(1, optimizationDegree+1)]
        
        #bStep is extremely expensive in runtime, doesn't need to be as precise as w
        bRangeMult, bStep = 5, 5
        
        return stepSizes, bRangeMult, bStep
    
    def fit(self, data, optimizationDegree=3):
        self.data = data
        # {||w||: [w, b]}
        optimizationDict = {}

        
        self.setFeatureExtremes()
        stepSizes, bRangeMult, bStep = self.setFitSteps(optimizationDegree)
        
        lastOptimal = self.featureMax * 10 #w is initially [lastOptimal, lastOptimal]
        
        for wStep in stepSizes:
            w = np.array([lastOptimal, lastOptimal]) #initialize vector to current best option
            optimized = False #allowed because it is a convex optimization
            self.optimize(optimized, bRangeMult, bStep, wStep, optimizationDict, w)
            
            magnitudes = sorted([n for n in optimizationDict])#get lowest w vector 
            optimalChoice = optimizationDict[magnitudes[0]]
            self.w, self.b = optimalChoice
            lastOptimal = optimalChoice[0][0] + wStep * 2 #minimum is within last two steps
        for i in self.data:
            for j in self.data[i]:
                print(np.dot(j, self.w) + self.b)
                    
    def optimize(self, optimized, bRangeMult, bStep, wStep, optimizationDict, w):
        vectorTransforms = [[1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1]]        
        while not optimized:
                for b in np.arange(-self.featureMax*bRangeMult, self.featureMax*bRangeMult, wStep*bStep):
                    for trans in vectorTransforms:
                        wTrans = trans * w
                        foundOption = True
                        #weakest link in the SVM fundamentally
                        #yi(xi dot w + B)>=1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(wTrans, xi) + b) >=1:#make sure every sample fits
                                    foundOption = False
                                    break
                            if not foundOption:
                                break
                        if foundOption:
                            optimizationDict[np.linalg.norm(wTrans)] = [wTrans, b]
                #reach minimum w value with current step
                if w[0] < 0:
                    optimized = True
                else:
                    w = w - wStep
                     
    def setFeatureExtremes(self):
        allFeatureValues = []
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
        label = np.sign(dot + self.b)
        
        if label !=0 and self.visualtion:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[label])
        return label
    
    def visualize(self):
        for i in self.data:
            for x in self.data[i]:
                self.ax.scatter(x[0], x[1], s=100, c=self.colors[i])
            
        #hyperplane = x.w + b
        #want: v=x.w+b
        #pos. SV--> v=1
        #neg. SV--> v=-1
        #decision plane --> v=0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]
        
        
        dataRange = (self.featureMin * 0.9, self.featureMax * 1.1)
        hypXMin = dataRange[0]
        hypXMax = dataRange[1]
                
        for value in [-1, 0, 1]:
            a = hyperplane(hypXMin, self.w, self.b, v=value)
            b = hyperplane(hypXMax, self.w, self.b, v=value)
            self.ax.plot([hypXMin, hypXMax], [a, b], c=self.colors[value])
        
        pyplot.show()

svm = SupportVectorMachine()
data = {-1: np.array([[1, 7],
                      [2, 8],
                      [3, 8]]),
        1: np.array([[5, 1],
                     [6, -1],
                     [7, 3]])}

svm.fit(data, optimizationDegree=3)
predcitionSets = [[0, 10],
                  [1, 3],
                  [3, 4],
                  [3, 5],
                  [5, 5],
                  [5, 6],
                  [6, -5],
                  [5, 8]]

for p in predcitionSets:
    svm.predict(p)
#svm.visualize()

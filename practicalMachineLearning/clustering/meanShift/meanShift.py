import numpy as np
from matplotlib import pyplot, style
style.use('fivethirtyeight')

class MeanShift:
    def __init__(self, radius=None, radiusNormStep=100): #bandwith is used synonymously with radius
        self.radius = radius
        self.radiusNormStep = radiusNormStep
    
    def initCentroids(self, data):
        centroids = {}
        for id in range(len(data)):
            centroids[id] = data[id]
        return centroids
    
    def getNewCentroids(self, data, weights, centroids):
        newCents = []
        
        for cent in centroids:
                inRadius = []
                cent = centroids[cent]
                
                for feats in data:
                    distance = np.linalg.norm(feats - cent)
                    if distance == 0:
                        distance = 0.00001
                    weightIndex = int(distance / self.radius)
                    if weightIndex > self.radiusNormStep - 1: #if beyond max, set to max
                        weightIndex = self.radiusNormStep - 1
                    toAdd = (weights[weightIndex] ** 2) * [feats]
                    inRadius += toAdd
                    
                newCentroid = np.average(inRadius, axis=0)#need axis=0
                newCents.append(tuple(newCentroid))
            
        return newCents
    
    def getUniqCentroids(self, centroids):
        uniqCentroids = sorted(list(set(centroids)))
        toPop = []
        
        for i in uniqCentroids:
            for j in uniqCentroids:
                if i == j:
                    pass
                elif np.linalg.norm(np.array(i) - np.array(j)) <= self.radius:
                    toPop.append(j)
                    break
        
        for c in toPop:
            try:
                uniqCentroids.remove(c)
            except ValueError:
                pass
        
        return uniqCentroids
    def setDynamicRadius(self, data):
        centerOfAll = np.average(data, axis=0)
        allNorm = np.linalg.norm(centerOfAll)
        self.radius = allNorm / self.radiusNormStep
        
    def fit(self, data):
        if self.radius == None:
            self.setDynamicRadius(data)
        
        centroids = self.initCentroids(data)
        weights = [i for i in range(self.radiusNormStep)][::-1] #[::-1] to reverse the list
        optimized = False
        
        while not optimized:
            newCents = self.getNewCentroids(data, weights, centroids)
            uniqCentroids = self.getUniqCentroids(newCents)
            oldCentroids = dict(centroids)
            
            centroids = {}
            for i in range(len(uniqCentroids)):
                centroids[i] = np.array(uniqCentroids[i])
            
            optimized = True
            for i in centroids:
                if not np.array_equal(oldCentroids[i], centroids[i]):
                    optimized = False
                    break
        self.centroids = centroids
        self.classifications = {}
        
        for i in range(len(self.centroids)):
            self.classifications[i] = []
        for feats in data:
            classif = self.predict(feats)
            self.classifications[classif].append(feats)
        
    def predict(self, data):
        distances = []
        for c in self.centroids:
            distances.append(np.linalg.norm(data - self.centroids[c]))
        classif = distances.index(min(distances))
        
        return classif


colors = 10* ['r', 'b', 'k', 'g', 'c', 'o']
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, .6],
              [9, 11], 
              [8, 2], 
              [10, 2],
              [9, 3]])
clf = MeanShift()
clf.fit(X)
centroids = clf.centroids

pyplot.scatter(X[:,0], X[:,1], s=100)
print(centroids)
for c in centroids:
    pyplot.scatter(centroids[c][0], centroids[c][1], c='k', s=100, marker='*')
    
pyplot.show()

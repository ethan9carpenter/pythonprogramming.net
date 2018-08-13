import numpy as np
from matplotlib import pyplot, style

style.use('fivethirtyeight')

class KMeans:
    def __init__(self, k=2, tol=0.001, maxIter=300):
        self.k = k
        self.tol = tol
        self.maxIter = maxIter
        
    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]
            
        for i in range(self.maxIter):
            self.classifcations = {}#setup classification map
            for i in range(self.k):
                self.classifcations[i] = []
            
            for feats in data:
                distances = []
                for cent in self.centroids:#calculate distances
                    distances.append(np.linalg.norm(feats - self.centroids[cent]))
                classif = distances.index(min(distances))#classify as closest centroid
                self.classifcations[classif].append(feats)#add to list of that classification
            
            lastCentroids = dict(self.centroids) #because of heap
            for i in range(self.k):
                self.centroids[i] = np.mean(self.classifcations[i], axis=0)
            
            optimized = True
            for c in self.centroids:
                pctChange = np.sum((self.centroids[c] - lastCentroids[i]) / lastCentroids[i])
                if pctChange > self.tol:
                    optimized = False
                    break
            if optimized:
                break
                    
    
    def predict(self, data):
        distances = []
        for i in self.centroids:
            distances.append(np.linalg.norm(data - self.centroids[i]))
        minClassif = distances.index(min(distances))
        
        return minClassif
        
colors = ['r', 'b', 'k', 'g', 'c', 'o']
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, .6],
              [9, 11], [1, 3], 
                     [8, 9],
                     [0, 3],
                     [5, 4],
                     [6, 4]])
clf = KMeans()
clf.fit(X)

for centroid in clf.centroids:
    pyplot.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], s=200, c='k')

for classif in clf.classifcations:
    color = colors[classif]
    for feats in clf.classifcations[classif]:
        pyplot.scatter(feats[0], feats[1], c=color, s=100, marker='x')
        
unknowns = np.array([[1, 3], 
                     [8, 9],
                     [0, 3],
                     [5, 4],
                     [6, 4]])
for feat in unknowns:
    color = colors[clf.predict(feat)]
    pyplot.scatter(feat[0], feat[1], marker='*', c=color, s=100)
    
pyplot.show()


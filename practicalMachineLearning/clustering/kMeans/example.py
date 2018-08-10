from matplotlib import style, pyplot
import numpy as np
from sklearn.cluster import KMeans

style.use('fivethirtyeight')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, .6],
              [9, 11]])
'''pyplot.scatter(X[:,0], X[:,1], s=150)
pyplot.show()'''

clf = KMeans(n_clusters=2)#default is 8
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ['r', 'b', 'k', 'g', 'c', 'o']

for i, coord in enumerate(X):
    pyplot.scatter(coord[0], coord[1], c=colors[labels[i]], s=50)

pyplot.scatter(centroids[:,0], centroids[:,1], marker='x', s=150)    
pyplot.show()

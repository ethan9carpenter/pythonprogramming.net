from sklearn.cluster import MeanShift
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style, pyplot
from sklearn.datasets.samples_generator import make_blobs
style.use('fivethirtyeight')

centers = [[1, 1, 1],
           [5, 5, 5], 
           [3, 10, 10]]
X = make_blobs(n_samples=1000, centers=centers, cluster_std=1)[0]

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
clusterCenters = ms.cluster_centers_
print(clusterCenters)
numClusters = len(np.unique(labels))

colors = ['r', 'g', 'b', 'o', 'c', 'k', 'y', 'm']
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]])
    
ax.scatter(clusterCenters[:,0], clusterCenters[:,1], clusterCenters[:,2], 
           marker='x', color='k', s=150, zorder=0)

pyplot.show()
import numpy as np
from sklearn.cluster import MeanShift
import pandas as pd
from sklearn import preprocessing
from clustering.kMeans.titanicExample import handleNonNum
from collections import Counter

df = pd.read_csv('titanic.csv')
originalDF = df.copy()

df.drop(['name', 'body'], inplace=True, axis=1)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
df = handleNonNum(df)

X = np.array(df.drop(['survived'], axis=1).astype(dtype=float))
X = preprocessing.scale(X) #Very important, especially in clustering, neighbors, etc
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
centers = clf.cluster_centers_
numClusters = len(np.unique(labels))
originalDF['clusterGroup'] = np.nan

for i in range(len(X)):
    originalDF.at[i, 'clusterGroup'] =  labels[i]#use iloc to reference the i-th row

survivalRates = {}

for i in range(numClusters):
    temp = originalDF[originalDF['clusterGroup'] == float(i)]  #all of the df where the cluster is i
    survived = temp[temp['survived'] == 1]
    rate = len(survived) / len(temp)
    survivalRates[i] = rate
for i in np.unique(labels):
    print(originalDF[originalDF['clusterGroup'] == i].describe()[['pclass', 'survived']])

print(Counter(labels))
print(survivalRates)


from matplotlib import pyplot, style
import numpy as np
from sklearn.cluster import KMeans
style.use('fivethirtyeight')
import pandas as pd
from sklearn import preprocessing

def handleNonNum(df):
    #convert objects to ints
    for column in df.columns.values:
        textValues = {}
        
        def convert(text):
            return textValues[text]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            contents = df[column].values.tolist()
            unique = set(contents)
            
            for i, uni in enumerate(unique):
                if uni not in textValues:
                    textValues[uni] = i
            
            df[column] = list(map(convert, df[column]))
    return df

cols = ['pclass', 'survived', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket',
       'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest']

df = pd.read_csv('titanic.csv')
df.drop(['name', 'body'], inplace=True, axis=1)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
df = handleNonNum(df)

featureCols = ['sex', 'age', 'sibsp', 'ticket', 'cabin', 'embarked', 'boat', 'home.dest']

toDrop = ['pclass', 'fare', 'parch']

'''for feature in featureCols:
    X = np.array(df.drop(['survived', feature] + toDrop, axis=1).astype(dtype=float))
    X = preprocessing.scale(X) #Very important, especially in clustering, neighbors, etc
    y = np.array(df['survived'])
    
    clf = KMeans(n_clusters=2)
    clf.fit(X)
    
    numCorrect = 0
    
    for a, b in zip(clf.labels_, df['survived']):
        if a == b:
            numCorrect += 1
    accuracy = numCorrect / len(X)
    accuracy = max(accuracy, 1-accuracy)
    print(accuracy, feature)'''
X = np.array(df.drop(['survived'] + toDrop, axis=1).astype(dtype=float))
X = preprocessing.scale(X) #Very important, especially in clustering, neighbors, etc
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

numCorrect = 0

for a, b in zip(clf.labels_, df['survived']):
    if a == b:
        numCorrect += 1
accuracy = numCorrect / len(X)
accuracy = max(accuracy, 1-accuracy)
print(accuracy)

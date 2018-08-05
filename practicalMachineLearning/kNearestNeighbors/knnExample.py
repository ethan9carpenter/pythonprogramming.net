import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 

df = pd.read_csv('resources/breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)

#===============================================================================
# drop useless data
# if we keep it it as a feature, it will be used in measuring 
# for nearest neighbor and render model useless
#===============================================================================
df.drop('id', inplace=True, axis=1)

X = np.array(df.drop('class', axis=1))
y = np.array(df['class'])

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=.2)

model = KNeighborsClassifier()
model.fit(X, y)

accuracy = model.score(xTest, yTest) #NOT CONFIDENCE (confidence is determined by vote %)
print(accuracy)

fake = np.array([4, 3, 2, 1, 9, 9, 9, 4, 3])
fake = fake.reshape(len(fake), -1)
prediction = model.predict(fake)

print(prediction)


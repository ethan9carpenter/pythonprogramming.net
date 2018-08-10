import pandas as pd
import numpy as np
from sklearn.svm import SVC
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

for c in [1, 2, 100, .1, .001, 100000000]:
    #===========================================================================
    # C determines how much slack is allowed to prevent overfitting, higher C-->more slack
    #===========================================================================
    model = SVC(C=c)
    model.fit(xTrain, yTrain)
    
    accuracy = model.score(xTest, yTest) #NOT CONFIDENCE (confidence is determined by vote %)
    print("C-{}:".format(c), accuracy)

#===============================================================================
# fake = np.array([4, 3, 2, 1, 9, 9, 9, 4, 3])
# fake = fake.reshape(len(fake), -1)
# prediction = model.predict(fake)
# 
# print(prediction)
#===============================================================================



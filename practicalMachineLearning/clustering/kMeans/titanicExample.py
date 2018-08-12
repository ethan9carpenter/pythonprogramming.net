from matplotlib import pyplot, style
import numpy as np
from sklearn.cluster import KMeans
style.use('fivethirtyeight')
import pandas as pd
from sklearn import preprocessing, model_selection

def handleNonNum(df):
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

df = pd.read_excel('titanic.xls')
df.drop(['name', 'body'], inplace=True, axis=1)
df.convert_objects(convert_numeric=True)
df.dropna(inplace=True)
df = handleNonNum(df)

print(df.head())

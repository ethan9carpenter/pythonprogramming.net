from matplotlib import pyplot, style
import numpy as np
from sklearn.cluster import KMeans
style.use('fivethirtyeight')
import pandas as pd
from sklearn import preprocessing, cross_validation

df = pd.read_excel('titanic.xls')

print(df.head())

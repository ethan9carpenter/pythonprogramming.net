import pandas as pd
import matplotlib.pyplot as pyplot
from matplotlib import style

style.use('fivethirtyeight')

height = {'meters': [10.26, 10.31, 10.27, 10.22, 10.23, 6212.42, 10.28, 10.25, 10.31]}
data = pd.DataFrame(height)
data['std'] = data['meters'].rolling(2).std()

data = data[data['std'] < data.describe()['meters']['std']]

print(data)

data['meters'].plot()
pyplot.show()












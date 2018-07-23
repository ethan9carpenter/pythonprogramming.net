import pandas as pd
import matplotlib.pyplot as pyplot
import matplotlib.style as style
from part7 import hpiBenchmark

style.use('fivethirtyeight')

#DATA
data = pd.read_pickle('resources/fiftyStates2.pickle')
data['TX1yr'] = data['TX'].resample('A').mean()

#data.dropna(inplace=True)
data.fillna(inplace=True, method='ffill')

#PLOT
fig = pyplot.figure()
ax1 = pyplot.subplot2grid((1, 1), (0, 0))
pyplot.legend().remove()

data['TX'].plot(ax=ax1)
data['TX1yr'].plot(color='k',ax=ax1)

pyplot.show()







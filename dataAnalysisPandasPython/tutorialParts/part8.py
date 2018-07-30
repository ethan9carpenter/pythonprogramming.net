import pandas as pd
import matplotlib.pyplot as pyplot
import matplotlib.style as style
from part7 import hpiBenchmark

style.use('fivethirtyeight')


data = pd.read_pickle('resources/fiftyStates2.pickle')
benchmark = hpiBenchmark()

fig = pyplot.figure()
ax1 = pyplot.subplot2grid((1, 1), (0, 0))
    
hpiCorr = data.corr()

TX1yr = data['TX'].resample('A').ohlc()

data['TX'].plot(ax=ax1)
TX1yr.plot(ax=ax1, color='k')

pyplot.legend().remove()
pyplot.show()

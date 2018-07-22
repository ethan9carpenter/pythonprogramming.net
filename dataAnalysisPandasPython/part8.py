import pandas as pd
import matplotlib.pyplot as pyplot
import matplotlib.style as style
from part7 import hpiBenchmark

data = pd.read_pickle('resources/fiftyStates2.pickle')
style.use('fivethirtyeight')

benchmark = hpiBenchmark()

def plotData():
    fig = pyplot.figure()
    ax1 = pyplot.subplot2grid((1, 1), (0, 0))
    
    data.plot(ax=ax1)
    benchmark.plot(color='k', ax=ax1, linewidth=10)
    
    pyplot.legend().remove()
    pyplot.show()
    
hpiCorr = data.corr()
print(hpiCorr.describe())   
import pandas as pd
from matplotlib import style, pyplot
import datetime as dt
import pandas_datareader.data as web

style.use('ggplot')

start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()

data = pd.read_csv('resources/tsla.csv', parse_dates=True, index_col=0)

data['Close'].plot()
pyplot.show()











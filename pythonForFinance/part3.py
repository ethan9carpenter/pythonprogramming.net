import pandas as pd
from matplotlib import style, pyplot
from mpl_finance import candlestick_ochl
import datetime as dt
import pandas_datareader.data as web

style.use('ggplot')

start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()

data = pd.read_csv('resources/tsla.csv', parse_dates=True, index_col=0)
data['100ma'] = data['Close'].rolling(100, min_periods=0).mean()
#data.dropna(inplace=True)

ax1 = pyplot.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
ax2 = pyplot.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)

ax1.plot(data.index, data['Close'])
ax1.plot(data.index, data['100ma'])
ax2.plot(data.index, data['Volume'])

pyplot.show()










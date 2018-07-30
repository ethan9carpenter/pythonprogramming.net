import pandas as pd
from matplotlib import style, pyplot, dates
from mpl_finance import candlestick_ohlc

style.use('ggplot')

data = pd.read_csv('resources/tsla.csv', parse_dates=True, index_col=0)
data['100ma'] = data['Close'].rolling(100, min_periods=0).mean()
#data.dropna(inplace=True)

ohlc = data['Close'].resample('10D').ohlc()
volume = data['Volume'].resample('10D').sum()

ohlc.reset_index(inplace=True)
ohlc['Date'] = ohlc['Date'].map(dates.date2num)

fig = pyplot.figure(figsize=(12, 7))

ax1 = pyplot.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
ax2 = pyplot.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, ohlc.values, width=2, colorup='g')
ax2.fill_between(volume.index.map(dates.date2num), volume.values, 0)# fill between volume.values and 0

pyplot.show()










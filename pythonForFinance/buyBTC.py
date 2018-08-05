import pandas as pd
from collections import OrderedDict
import pytz
from zipline.api import symbol, set_benchmark, order_target_percent, get_open_orders, record
from zipline import run_algorithm
from datetime import datetime
from trading_calendars.trading_calendar import TradingCalendar
from datetime import time
from pandas.tseries.offsets import CustomBusinessDay
from pytz import timezone
from trading_calendars.utils.memoize import lazyval


def initialize(context):
    set_benchmark(symbol("BTC"))

def handle_data(context, data):

    slowma = data.history(symbol("BTC"), fields='price', bar_count=50, frequency='1m').mean()
    fastma = data.history(symbol("BTC"), fields='price', bar_count=10, frequency='1m').mean()

    if fastma < slowma:
        if symbol("BTC") not in get_open_orders():
            order_target_percent(symbol("BTC"), 0.04)

    if fastma > slowma:
        if symbol("BTC") not in get_open_orders():
            order_target_percent(symbol("BTC"), 0.96)

    record(BTC=data.current(symbol('BTC'), fields='price'))

data = OrderedDict()
data['BTC'] = pd.read_csv("resources/BTC-USD.csv")

data['BTC']['date'] = pd.to_datetime(data['BTC']['time'], unit='s', utc=True)
data['BTC'].set_index('date', inplace=True)
data['BTC'].drop('time', axis=1, inplace=True)
data['BTC'] = data['BTC'].resample("1min").mean()
data['BTC'].fillna(method="ffill", inplace=True)
data['BTC'] = data['BTC'][["low","high","open","close","volume"]]
#print(data['BTC'].head())

panel = pd.Panel(data)
panel.minor_axis = ["low","high","open","close","volume"]
panel.major_axis = panel.major_axis.tz_convert(pytz.utc)

class TwentyFourHR(TradingCalendar):
    @property
    def name(self):
        return "twentyfourhr"
    @property
    def tz(self):
        return timezone("UTC")
    @property
    def open_time(self):
        return time(0, 0)
    @property
    def close_time(self):
        return time(23, 59)
    @lazyval
    def day(self):
        return CustomBusinessDay(
            weekmask='Mon Tue Wed Thu Fri Sat Sun',
        )

perf = run_algorithm(start = datetime(2018, 3, 25, 0, 0, 0, 0, pytz.utc),
                      end = datetime(2018, 3, 26, 0, 0, 0, 0, pytz.utc),
                      initialize = initialize,
                      trading_calendar = TwentyFourHR(),
                      capital_base = 10000,
                      handle_data = handle_data,
                      data_frequency = 'minute',
                      data = panel)

print(perf)
import pandas as pd
from collections import OrderedDict
import pytz
from zipline.api import order, record, symbol, set_benchmark
from zipline import run_algorithm
from datetime import datetime
from matplotlib import pyplot

filePath = 'resources/SPY.csv'
data = OrderedDict()

data['SPY'] = pd.read_csv(filePath, index_col=0, parse_dates=True)
data['SPY'] = data['SPY'][['open', 'high', 'low', 'close', 'volume']]

panel = pd.Panel(data)
panel.minor_axis = ['open', 'high', 'low', 'close', 'volume']
panel.major_axis = panel.major_axis.tz_localize(pytz.utc)


def intialize(context):
    context.stock = symbol('SPY')
    set_benchmark(context.stock)

def handleData(context, data):
    order(context.stock, 10)
    record(AAPL=data.current(context.stock, 'price'))
    
def analyze(context, perf):
    perf['portfolio_value'].plot()
    pyplot.show()
    
result = run_algorithm(start = datetime(2010, 1, 1, 0, 0, 0, 0, pytz.utc),
                       end = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc), 
                       initialize = intialize, 
                       capital_base = 10000000, 
                       handle_data = handleData,
                       analyze = analyze,
                       data = panel)


    

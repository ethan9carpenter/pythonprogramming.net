from zipline.api import order, record, symbol
from zipline import run_algorithm
import datetime as dt
import pickle
import pytz
from matplotlib import pyplot, style

def intialize(context):
    pass

def handleData(context, data):
    stock = symbol('AAPL')
    
    order(stock, 10)
    record(AAPL=data.current(stock, 'price'))
    
def analyze(context, perf):
    style.use('ggplot')
    
    figure, (ax1, ax2) = pyplot.subplots(nrows=2, ncols=1, sharex=True)
    figure.set_size_inches(16, 10)
    
    perf['portfolio_value'].plot(ax=ax1)
    perf['AAPL'].plot(ax=ax2)
    
    ax1.set_ylabel('Portfolio Value')
    ax2.set_ylabel('AAPL')
    
    pyplot.legend()
    
    pyplot.show()
    
    with open('backtests/aaplAlgoTest2.pickle', 'wb') as file:
        pickle.dump(perf, file)


start = dt.datetime(2010, 1, 1, tzinfo=pytz.utc)
end = dt.datetime(2018, 1, 1, tzinfo=pytz.utc)

df = run_algorithm(start = start,
                   end = end,
                   initialize = intialize,
                   handle_data=handleData,
                   capital_base = 1000000,
                   #analyze = analyze,
                   bundle = 'quantopian-quandl')
print(df.head())
df[['portfolio_value', 'AAPL']].pct_change().fillna(0).add(1).cumprod().sub(1).plot()
pyplot.show()





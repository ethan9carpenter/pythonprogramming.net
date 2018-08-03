from zipline.api import order, record, symbol
from zipline import run_algorithm
import datetime as dt
import pickle
from pytz import timezone

def intialize(context):
    pass

def handleData(context, data):
    stock = symbol('AAPL')
    
    order(stock, 10)
    record(AAPL=data.current(stock, 'price'))
    

    
start = dt.datetime(2010, 1, 1, tzinfo=None)
end = dt.datetime(2018, 1, 1, tzinfo=None)

est = timezone('US/Eastern')

start = est.localize(start)
end = est.localize(end)

df = run_algorithm(start = start,
                   end = end,
                   initialize = intialize,
                   handle_data=handleData,
                   capital_base = 1000000,
                   bundle = 'quantopian-quandl')
print(df.head())

with open('aaplAlgoTest.pickle', 'wb') as file:
    pickle.dump(df, file)

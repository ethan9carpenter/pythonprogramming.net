import pandas as pd
from matplotlib import pyplot, style

style.use('ggplot')

results = pd.read_pickle('backtests/aaplAlgoTest.pickle')

figure, (ax1, ax2) = pyplot.subplots(nrows=2, ncols=1, sharex=True)
figure.set_size_inches(16, 10)

results['portfolio_value'].plot(ax=ax1)
results['AAPL'].plot(ax=ax2)


pyplot.show()
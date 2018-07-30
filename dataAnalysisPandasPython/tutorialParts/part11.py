import pandas as pd
import matplotlib.pyplot as pyplot
import matplotlib.style as style

style.use('fivethirtyeight')

#DATA
data = pd.read_pickle('resources/fiftyStates2.pickle')
#data['TX12MA'] = data['TX'].rolling(12).mean()
#data['TX12STD'] = data['TX'].rolling(12).std()

#PLOT
fig = pyplot.figure(figsize=(11, 7))

ax1 = pyplot.subplot2grid((2, 1), (0, 0))
ax2 = pyplot.subplot2grid((2, 1), (1, 0), sharex=ax1)

rollingCorr = data['TX'].rolling(12).corr(other=data['AK'])

data['TX'].plot(ax=ax1, label='TX HPI')
data['AK'].plot(ax=ax1, label='AK HPI')


ax1.legend(loc=4)
rollingCorr.plot(ax=ax2)

pyplot.show()







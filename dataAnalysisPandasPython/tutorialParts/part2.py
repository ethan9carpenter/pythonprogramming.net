import pandas as pd
import matplotlib.pyplot as pyplot
from matplotlib import style

stats = {'Day': list(range(1, 7)),
         'Visitors': [43, 34, 65, 56, 29, 76],
         'Bounce Rate': [65, 67, 78, 65, 45, 52]}

df = pd.DataFrame(stats)

df.set_index('Day', inplace=True)

print(df.Visitors)
print(df['Bounce Rate'])

df.plot()
pyplot.show()

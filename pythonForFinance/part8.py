import pandas as pd
from matplotlib import style, pyplot
import numpy as np

style.use('ggplot')

def visualizeData():
    df = pd.read_csv('resources/mergedSP500.csv')
    dfCorr = df.corr()
    
    #dfCorr.to_csv('resources/sp500corr.csv')
    
    corrData = dfCorr.values
    fig1 = pyplot.figure(figsize=(11, 7))
    ax1 = fig1.add_subplot(111)
    heatmap1 = ax1.pcolor(corrData, cmap=pyplot.cm.RdYlGn)
    fig1.colorbar(heatmap1)
    
    ax1.set_xticks(np.arange(corrData.shape[1]) + .5, minor=False)
    ax1.set_yticks(np.arange(corrData.shape[0]) + .5, minor=False)
    
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    
    columnLabels = dfCorr.columns
    rowLabels = dfCorr.index
    ax1.set_xticklabels(columnLabels)
    ax1.set_yticklabels(rowLabels)

    pyplot.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    pyplot.tight_layout()
    #pyplot.savefig('resources/correlationMap.svg', dpi=(300))
    pyplot.show()
    
visualizeData()
    























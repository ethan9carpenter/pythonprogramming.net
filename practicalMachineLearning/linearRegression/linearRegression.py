from statistics import mean
import numpy as np
from matplotlib import pyplot, style
import random

class LinearRegression():
    def __init__(self): 
        style.use('fivethirtyeight')       
        self.rSquared = None
        self.intercept = None
        self.slope = None
    
    def fit(self, xData, yData):
        self.setData(xData, yData)
        self.slope = self.calculateSlope()
        self.intercept = self.calculateIntercept()
        
    def setData(self, xData, yData):
        self.xData = xData.astype(float)
        self.yData = yData.astype(float)
    
    def calculateSlope(self):
        meanX = mean(self.xData)
        meanY = mean(self.yData)
        meanXsquared = mean(self.xData * self.xData)
        meanXY = mean(self.xData * self.yData)
        
        slope = (meanX * meanY - meanXY) / (meanX ** 2 - meanXsquared)
        
        return slope
    
    def calculateIntercept(self):
        return mean(self.yData) - self.slope * mean(self.xData)
    
    def predict(self, xData):
        return xData * self.slope + self.intercept
    
    def score(self):
        #Returns r^2
        sqErrorReg = (self.yData - self.predict(self.xData)) ** 2
        sqErrorData = (self.yData - mean(self.yData)) ** 2
        
        self.rSquared = 1 - mean(sqErrorReg) / mean(sqErrorData)
        
        return self.rSquared
    
    def __str__(self):
        return ('y=' + '{0:.4}'.format(str(self.slope)) 
                + "x+" + '{0:.4}'.format(str(self.intercept)))
        
    def plot(self):
        text = str(self) + '\nCorrelation: ' + '{0:.5}'.format(str(self.score()))
        
        ax = pyplot.figure().add_subplot(111)
        
        pyplot.scatter(self.xData, self.yData)
        regY = self.predict(self.xData)
        pyplot.plot(self.xData, regY, c='red')
        
        ax.text(0, max(self.yData), text)
        
        pyplot.show()

def buildDataset(size, variance, step=1, direction=False):
    if direction and direction == 'positive':
        step = abs(step)
    elif direction and direction == 'negative':
        step = -abs(step)
    
    yData = []
    xData = [i for i in range(size)]
    
    for i in range(size):
        try:
            y = i * step + random.randrange(-variance, variance)
        except ValueError:
            y = i * step
        yData.append(y)
    
    return np.array(xData), np.array(yData)
    
x, y = buildDataset(40, 0, direction='positive')

line = LinearRegression()
line.fit(x, y)
line.plot()
from statistics import mean
import numpy as np

class LinearRegression():
    def __init__(self):        
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
    
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 6, 5, 6])

line = LinearRegression()
line.fit(x, y)
print(line.slope, line.intercept, line.score())
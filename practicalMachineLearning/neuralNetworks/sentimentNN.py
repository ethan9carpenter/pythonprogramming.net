import tensorflow as tf
from processSentimentData import createFeaturesAndLabels
import numpy as np

"""
input --> weight --> hidden layer 1 (activation functions) --> ... --> 
weight --> hidden layer n (activation function) --> weights --> output (feed forward)

compare to intended output with a cost function

optimizer will try to minimize cost (backpropogation), going backwards from the
output and manipulates weights in order to minimize cost

feed forward + backpropogation = epoch (will repeat many times)
"""

#one_hot=True means that each feature set can only be one classification
xTrain, yTrain, xTest, yTest = createFeaturesAndLabels(pos='res/pos.txt', neg='res/neg.txt')

numNodes = [500] * 3

featuresLength = len(xTrain[0])
numClasses = 2
batchSize = 100 #to deal with extremely large datasets

#matrix = height x weight, if we add the parameter TF will throw an 
#error when it encounters something of a different shape
x = tf.placeholder('float', [None, featuresLength])
y = tf.placeholder('float')

def networkModel(data):
    layers, outputLayer = createLayers()
    tfLayers = createTFLayers(layers, data)
    output = tf.matmul(tfLayers[-1], outputLayer['weights']) + outputLayer['biases']
    
    return output

def createTFLayers(layers, data):
    tfLayers = []
    
    for i in range(len(layers)):
        weights = layers[i]['weights']
        biases = layers[i]['biases']
        
        if i == 0:
            lay = tf.add(tf.matmul(data, weights), biases)
        else:
            lay = tf.add(tf.matmul(tfLayers[-1], weights), biases)
        lay = tf.nn.relu(lay)
        tfLayers.append(lay)
            
    return tfLayers

def createLayers():
    layers = []
    
    for i in range(len(numNodes)):
            # (input data * weights) + bias, causes network to be more 
            # dynamic and in the case of uniform data
            nNodes = numNodes[i]
            
            if i == 0:
                layer = {'weights': tf.Variable(tf.random_normal([featuresLength, nNodes])),
                         'biases': tf.Variable(tf.random_normal([nNodes]))}
            else:
                prevNodes = numNodes[i-1]
                layer = {'weights': tf.Variable(tf.random_normal([prevNodes, nNodes])),
                         'biases': tf.Variable(tf.random_normal([nNodes]))}
            layers.append(layer)
    
    outputLayer = {'weights': tf.Variable(tf.random_normal([numNodes[-1], numClasses])),
                  'biases': tf.Variable(tf.random_normal([numClasses]))}
    
    return layers, outputLayer

def train(x, numEpochs=3):
    prediction = networkModel(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        
        for epNum in range(numEpochs):
            epochLoss = 0
            
            for i in range(len(xTrain)):
                start = i
                end = i + batchSize
                epochX = np.array(xTrain[start:end])
                epochY = np.array(yTrain[start:end])
                
                _, c = session.run([optimizer, cost], feed_dict={x: epochX, y: epochY})
                epochLoss += c
            print(epNum+':', epochLoss)
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: xTest, y: yTest}))
    
train(x)
        
    
    
    
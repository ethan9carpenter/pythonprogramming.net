import tensorflow as tf
import numpy as np

def networkModel(X, numNodes):
    layers, outputLayer = createLayers(numNodes, len(X[0]))
    tfLayers = createTFLayers(layers, X)
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

def createLayers(numNodes, featLength):
    layers = []
    
    for i in range(len(numNodes)):
            # (input data * weights) + bias, causes network to be more 
            # dynamic and in the case of uniform data
            nNodes = numNodes[i]
            
            if i == 0:
                layer = {'weights': tf.Variable(tf.random_normal([featLength, nNodes])),
                         'biases': tf.Variable(tf.random_normal([nNodes]))}
            else:
                prevNodes = numNodes[i-1]
                layer = {'weights': tf.Variable(tf.random_normal([prevNodes, nNodes])),
                         'biases': tf.Variable(tf.random_normal([nNodes]))}
            layers.append(layer)
    
    outputLayer = {'weights': tf.Variable(tf.random_normal([numNodes[-1], numClasses])),
                  'biases': tf.Variable(tf.random_normal([numClasses]))}
    
    return layers, outputLayer

def train(X, numEpochs=3, numLayers=3, numNodes=500):
    x = tf.placeholder('float', [None, len(X[0])])
    y = tf.placeholder('float')
    
    numNodes = [numNodes] * numLayers
    prediction = networkModel(X)
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

xTrain, yTrain, xTest, yTest = []
numClasses = 2
batchSize = 100 #to deal with extremely large datasets

train(xTrain)
        
    
    
    

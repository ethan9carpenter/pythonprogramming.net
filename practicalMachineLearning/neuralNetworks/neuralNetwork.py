import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
input --> weight --> hidden layer 1 (activation functions) --> ... --> 
weight --> hidden layer n (activation function) --> weights --> output (feed forward)

compare to intended output with a cost function

optimizer will try to minimize cost (backpropogation), going backwards from the
output and manipulates weights in order to minimize cost

feed forward + backpropogation = epoch (will repeat many times)
"""

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
#one_hot=True means that each feature set can only be one classification

numNodes = [500] * 3

numClasses = 10
batchSize = 100 #to deal with extremely large datasets

#matrix = height x weight, if we add the parameter TF will throw an 
#error when it encounters something of a different shape
X = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def networkModel(data):
    layers, outputLayer = createLayers()
    tfLayers = createTFLayers(layers, data)
    output = tf.matmul(tfLayers[-1], outputLayer['weights']) + outputLayer['biases']
    
    return output

def createTFLayers(layers, data):
    tfLayers = []
    
    for i in range(len(layers)):
        weights, biases = layers[i]
        
        if i == 0:
            lay = tf.add(tf.matmul(data, weights) + biases)
        else:
            lay = tf.add(tf.matmul(tfLayers[-1], weights) + biases)
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
                layer = {'weights': tf.Variable(tf.random_normal([784, nNodes])),
                         'biases': tf.Variable(tf.random_normal(nNodes))}
            else:
                prevNodes = numNodes[i-1]
                layer = {'weights': tf.Variable(tf.random_normal([prevNodes, nNodes])),
                         'biases': tf.Variable(tf.random_normal(nNodes))}
            layers.append(layer)
    
    outputLayer = {'weights': tf.Variable(tf.random_normal([numNodes[-1], numClasses])),
                  'biases': tf.Variable(tf.random_normal(numClasses))}
    
    return layers, outputLayer
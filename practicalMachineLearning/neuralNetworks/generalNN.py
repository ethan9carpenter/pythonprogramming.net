import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



def networkModel(data, numNodes, numClasses):
    layers, outputLayer = createLayers(numNodes, numClasses)
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

def createLayers(numNodes, numClasses):
    layers = []
    
    for i in range(len(numNodes)):
            # (input data * weights) + bias, causes network to be more 
            # dynamic and in the case of uniform data
            nNodes = numNodes[i]
            
            if i == 0:
                layer = {'weights': tf.Variable(tf.random_normal([784, nNodes])),
                         'biases': tf.Variable(tf.random_normal([nNodes]))}
            else:
                prevNodes = numNodes[i-1]
                layer = {'weights': tf.Variable(tf.random_normal([prevNodes, nNodes])),
                         'biases': tf.Variable(tf.random_normal([nNodes]))}
            layers.append(layer)
    
    outputLayer = {'weights': tf.Variable(tf.random_normal([numNodes[-1], numClasses])),
                  'biases': tf.Variable(tf.random_normal([numClasses]))}
    
    return layers, outputLayer

def train(data, x, y, numClasses, numNodes=500, numLayers=3, batchSize=100, numExamples=None):
    if not numExamples:
        numExamples = len(data)
    
    numNodes = 3 * [500]
    prediction = networkModel(x, numNodes, numClasses)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    numEpochs = 5 #feed forward and backward propogation
        
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        
        for i in range(numEpochs):
            epochLoss = 0
            for _ in range(int(numExamples / batchSize)):
                epochX, epochY = data.next_batch(batchSize)
                _, c = session.run([optimizer, cost], feed_dict={x: epochX, y: epochY})
                epochLoss += c
            print(i, ':', epochLoss)
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
mnist = mnist.train

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

train(mnist, x, y, 10, numExamples=mnist.num_examples)
        
    
    
    
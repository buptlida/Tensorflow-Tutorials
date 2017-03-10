#!/usr/bin/env python

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data   #import input_data

batch_size = 128

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def learn_model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trainX = mnist.train.images
trainY = mnist.train.labels
testX  = mnist.test.images
testY  = mnist.test.labels

X = tf.placeholder("float", [None, 784]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])
w = init_weights([784, 10]) # like in linear regression, we need a shared variable weight matrix for logistic regression

#train graph
train_y  = learn_model(X, w)
cost     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_y, Y)) # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer

#test graph
test_y     = learn_model(X, w)
test_check = tf.equal(tf.argmax(test_y, 1), tf.argmax(testY, 1)) # at predict time, evaluate the argmax of the logistic regression
test_op    = tf.reduce_mean(tf.cast(test_check, "float"))

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    startRange = range(0, len(trainX), batch_size)
    endRange   = range(batch_size, len(trainX) + 1, batch_size)

    for i in range(100):
        for (start, end) in zip(startRange, endRange):
            sess.run(train_op, feed_dict={X: trainX[start:end], Y: trainY[start:end]})

        test_result = sess.run(test_op, feed_dict={X: testX})
        print(i, test_result)

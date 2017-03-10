#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data   #import input_data

batch_size = 128

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trainX = mnist.train.images
trainY = mnist.train.labels
testX  = mnist.test.images
testY  = mnist.test.labels

X   = tf.placeholder("float", [None, 784])
Y   = tf.placeholder("float", [None, 10])
w_h = init_weights([784, 625]) # create symbolic variables
w_o = init_weights([625, 10])

train_y    = model(X, w_h, w_o)
cost       = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_y, Y)) # compute costs
train_op   = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer

'''利用上面训练得到的权重矩阵w_h, w_o计算测试数据集的标签，然后和实际的标签进行比较'''
test_y     = model(X, w_h, w_o)
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

## AlexNet tensorflow

import tensorflow as tf
import numpy as np


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01),dtype = tf.float32)

def init_bias_zero(shape):
    return tf.Variable(tf.zeros(shape,dtype=tf.float32))

def init_bias_ones(shape):
    return tf.Variable(tf.ones(shape,dtype=tf.float32))



"""
input = [None , 224, 224, 3]   -
conv2d 11*11 - 4 - 96 - Relu   -  max_pool - 3*3 - 2 - LRN (local_response_normalization)
conv2d 5*5 - 1 - 256 - Relu    -  max_pool - 3*3 - 2 - LRN

conv2d - 3*3 - 1 - 384 - Relu
conv2d - 3*3 - 1 - 384 - Relu
conv2d - 3*3 - 1 - 256 - Relu  - max_pool - 3*3 - 2  - LRN

reshape
FC4096 - tanh - dropout 0.5
FC4096 - tanh - dropout 0.5
FC8 - softmax
"""

weights ={
"w1":init_weights([11,11,3,96]),
"w2":init_weights([5,5,96,256]),
"w3":init_weights([3,3,256,384]),
"w4":init_weights([3,3,384,384]),
"w5":init_weights([3,3,384,256]),
"w6":init_weights([4096,4096]),
"w7":init_weights([4096,4096]),
"w8":init_weights([4096,8])
}

bias ={
"b1":init_bias_zero([96]),
"b2":init_bias_ones([256]),
"b3":init_bias_zero([384]),
"b4":init_bias_ones([384]),
"b5":init_bias_ones([256]),
"b6":init_bias_ones([4096]),
"b7":init_bias_ones([4096])
}

def alexnet(X,weights,bias):
    network = tf.nn.conv2d(X,weights["w1"],strides=[1,4,4,1], padding = "SAME",name="conv1")
    network = tf.nn.relu(tf.nn.bias_add(network,bias["b1"]))
    network = tf.nn.max_pool(network,ksize=[1,3,3,1],strides=[1,2,2,1], padding = "SAME",name="max1")
    network = tf.nn.local_response_normalization(network, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name="lrn1")
    network = tf.nn.conv2d(network,weights["w2"],strides=[1,2,2,1],padding="SAME",name="conv2")
    network =tf.nn.relu(tf.nn.bias_add(network,bias["b2"]))
    network = tf.nn.max_pool(network,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="max2")
    network = tf.nn.local_response_normalization(network, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name="lrn2")
    network = tf.nn.conv2d(network,weights["w3"],strides=[1,1,1,1],padding="SAME",name="conv3")
    network = tf.nn.relu(tf.nn.bias_add(network,bias["b3"]))
    network = tf.nn.conv2d(network,weights["w4"],strides=[1,1,1,1],padding="SAME",name="conv4")
    network = tf.nn.relu(tf.nn.bias_add(network,bias["b4"]))
    network = tf.nn.conv2d(network,weights["w5"],strides=[1,1,1,1],padding="SAME",name="conv5")
    network = tf.nn.relu(tf.nn.bias_add(network,bias["b5"]))
    network = tf.nn.max_pool(network,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="max3")
    network = tf.nn.local_response_normalization(network, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name="lrn2")
    pool6Shape = network.get_shape().as_list()
    network = tf.reshape(network, [-1, pool6Shape[1] * pool6Shape[2] * pool6Shape[3]])
    network = tf.tanh(tf.add(tf.matmul(network, weights["w6"]), bias["b6"]))
    network = tf.nn.dropout(network,0.5)
    network = tf.tanh(tf.add(tf.matmul(network, weights["w7"]), bias["b7"]))
    network = tf.nn.dropout(network,0.5)
    network = tf.matmul(network,weights["w8"])
    return network # add a softmax logit layer when computing the cost

# sample model


""" Kaggle Fish """

#writing generator functions and other Image pre-processing functions

import tensorflow as tf
import numpy as np

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01),dtype = tf.float32)

def init_bias(shape):
    return tf.Variable(tf.zeros(shape,dtype=tf.float32))


def cnn_layer(input,weight,bias,name = "conv"):
    with tf.name_scope(name):
        x = tf.nn.conv2d(input,weight,strides=[1,2,2,1],padding= "VALID")
        x = tf.nn.relu(tf.nn.bias_add(x,bias))
        x = tf.nn.max_pool(x,ksize= [1,2,2,1], strides = [1,1,1,1], padding = "VALID")
    return x

def model(X,w1,w2,w3,w4,w5,w6,w7,w8,w9,b1,b2,b3,b4,b5,b6,b7,b8,b9):
    #layer1
     x = cnn_layer(X, w1, b1, name="conv1")
     x = cnn_layer(x, w2, b2, name="conv2")
     x = cnn_layer(x, w3, b3, name="conv3")
     x = cnn_layer(x, w4, b4, name="conv4")
     x = cnn_layer(x, w5, b5, name="conv5")
     x = cnn_layer(x, w6, b6, name="conv6")
     pool6Shape = x.get_shape().as_list()
     x = tf.reshape(x, [-1, pool6Shape[1] * pool6Shape[2] * pool6Shape[3]])
     x = tf.add(tf.matmul(x, w7), b7)
     x = tf.nn.relu(x)
     x = tf.nn.dropout(x,0.5)
     x = tf.add(tf.matmul(x, w8), b8)
     x = tf.nn.relu(x)
     x = tf.nn.dropout(x,0.5)
     x = tf.add(tf.matmul(x, w9), b9)
     return x

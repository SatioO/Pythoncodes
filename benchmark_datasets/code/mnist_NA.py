""" fractional max-pooling """

import tensorflow as tf
import numpy as np


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01),dtype = tf.float32)

def init_bias(shape):
    return tf.Variable(tf.zeros(shape,dtype=tf.float32))

# random overlapping - with 250 repitation

def cnn_layer(input,weight,bias,alpha = 0.01,name = "conv"):
    with tf.name_scope(name):
        x = tf.nn.conv2d(input,weight,strides=[1,1,1,1],padding= "SAME")
        x = tf.maximum(alpha*tf.nn.bias_add(x,bias),tf.nn.bias_add(x,bias))
        x = tf.nn.fractional_max_pool(x,pooling_ratio= [1,1.414,1.414,1],overlapping = True, seed = 0)
    return x[0]

# tf.nn.relu(tf.nn.bias_add(x,bias))
# tf.maximum(alpha*x,x

# Without Training set Agumentation or dropout
def model(X,weights,bias):
    #layer1
    x = cnn_layer(X, weights["w1"] , bias["b1"] ,name="conv1")
    x = cnn_layer(x, weights["w2"] , bias["b2"] ,name="conv2")
    x = cnn_layer(x, weights["w3"] , bias["b3"] ,name="conv3")
    x = cnn_layer(x, weights["w4"] , bias["b4"] ,name="conv4")
    x = cnn_layer(x, weights["w5"] , bias["b5"] ,name="conv5")
    x = cnn_layer(x, weights["w6"] , bias["b6"] ,name="conv6")
    x = tf.nn.conv2d(x,weights["w7"], strides=[1,1,1,1],padding="VALID")
    x = tf.maximum(0.01*tf.nn.bias_add(x,bias["b7"]),tf.nn.bias_add(x,bias["b7"]))
    x = tf.nn.conv2d(x,weights["w8"], strides=[1,1,1,1],padding="VALID")
    x = tf.maximum(0.01*tf.nn.bias_add(x,bias["b8"]),tf.nn.bias_add(x,bias["b8"]))
    pool6Shape = x.get_shape().as_list()
    x = tf.reshape(x,[-1,  pool6Shape[1]* pool6Shape[2] * pool6Shape[3]])
    x = tf.add(tf.matmul(x,weights["w9"]), bias["b9"])
    return x

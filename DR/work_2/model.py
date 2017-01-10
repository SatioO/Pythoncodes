# model building

#writing generator functions and other Image pre-processing functions

import tensorflow as tf
import numpy as np

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01),dtype = tf.float32)

def init_bias(shape):
    return tf.Variable(tf.zeros(shape,dtype=tf.float32))


def cnn_layer(input,weight,bias,name = "conv"):
    with tf.name_scope(name):
        x = tf.nn.conv2d(input,weight,strides=[1,1,1,1],padding= "SAME")
        x = tf.nn.relu(tf.nn.bias_add(x,bias))
        x = tf.nn.fractional_max_pool(x,[1,1.414,1.414,1],pseudo_random=True)
    return x[0]

"""

C16-3, FMP1.414, C16-3, FMP1.414
C32-3, FMP1.414, C32-3, FMP1.414
C64-3, FMP1.414, C64-3, FMP1.414
C128-3, FMP1.414, C128-3, FMP1.414
C192-3, FMP1.414, C192-3, FMP1.414
C256-3, FMP1.414, C256-3, FMP1.414
C384-3, FMP1.414, C384-3, FMP1.414
C512-3, FMP1.414, C512-3, FMP1.414
D(0.5), FC1024, FC1024

"""

def model(X,weights,bias):
    x = cnn_layer(X, weights["w1"], bias["b1"], name="conv1")
    x = cnn_layer(x, weights["w2"], bias["b2"], name="conv2")
    x = cnn_layer(x, weights["w3"], bias["b3"], name="conv3")
    x = cnn_layer(x, weights["w4"], bias["b4"], name="conv4")
    x = cnn_layer(x, weights["w5"], bias["b5"], name="conv5")
    x = cnn_layer(x, weights["w6"], bias["b6"], name="conv6")
    x = cnn_layer(x, weights["w7"], bias["b7"], name="conv7")
    x = cnn_layer(x, weights["w8"], bias["b8"], name="conv8")
    x = cnn_layer(x, weights["w9"], bias["b9"], name="conv9")
    x = cnn_layer(x, weights["w10"], bias["b10"], name="conv10")
    x = cnn_layer(x, weights["w11"], bias["b11"], name="conv11")
    x = cnn_layer(x, weights["w12"], bias["b12"], name="conv12")
    x = cnn_layer(x, weights["w13"], bias["b13"], name="conv13")
    x = cnn_layer(x, weights["w14"], bias["b14"], name="conv14")
    x = cnn_layer(x, weights["w15"], bias["b15"], name="conv15")
    x = cnn_layer(x, weights["w16"], bias["b16"], name="conv16")
    pool6Shape = x.get_shape().as_list()
    x = tf.reshape(x, [-1, pool6Shape[1] * pool6Shape[2] * pool6Shape[3]])
    x = tf.nn.dropout(x,0.5)
    x = tf.add(tf.matmul(x, weights["w17"]), bias["b17"]) # 512,1024
    x = tf.nn.relu(x)
    x = tf.add(tf.matmul(x, weights["w18"]), bias["b18"]) # 1024,1024
    x = tf.nn.relu(x)
    x = tf.matmul(x, weights["w19"]) # 1024,5
    return x

""" Kaggle Fish """

#writing generator functions and other Image pre-processing functions

"""

Possible Image pre-processing required
 - Reduce the Image size
 - Day and night effects
 - fish can be repersented in any way (360 degree rotations)
 - view points changing - zooming
 - ....


"""
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01),dtype = tf.float32)

def bias(shape):
    return tf.Variable(tf.zeros(shape,dtype=tf.float32))


def cnn_layer(input,weight,bias,name = "conv"):
    with tf.name_scope(name):
        x = tf.nn.conv2d(input,weight,strides=[1,2,2,1],padding= "VALID")
        x = tf.nn.relu(tf.nn.bias_add(x,bias))
        x = tf.nn.fractional_max_pool(x,pooling_ratio= [1,1.414,1.414,1])
    return x 

def model(X):
    #layer1
     x = cnn_layer(X, init_weights([3,3,3,16]), bias([16]), name="conv1")
     x = cnn_layer(x, init_weights([3,3,16,16]), bias([16]), name="conv2")
     x = cnn_layer(x, init_weights([3,3,16,32]), bias([32]), name="conv3")
     x = cnn_layer(x, init_weights([3,3,32,32]), bias([32]), name="conv4")
     x = cnn_layer(x, init_weights([3,3,32,64]), bias([64]), name="conv5")
     x = cnn_layer(x, init_weights([3,3,64,64]), bias([64]), name="conv6")
     x = cnn_layer(x, init_weights([3,3,64,128]), bias([128]), name="conv7")
     x = cnn_layer(x, init_weights([3,3,128,128]), bias([128]), name="conv8")
     x = cnn_layer(x, init_weights([3,3,128,192]), bias([192]), name="conv9")
     x = cnn_layer(x, init_weights([3,3,192,192]), bias([192]), name="conv10")
     x = cnn_layer(x, init_weights([3,3,192,256]), bias([256]), name="conv11")
     x = cnn_layer(x, init_weights([3,3,256,256]), bias([256]), name="conv12")
     x = cnn_layer(x, init_weights([3,3,256,384]), bias([384]), name="conv13")
     x = cnn_layer(x, init_weights([3,3,384,384]), bias([384]), name="conv14")
     x = cnn_layer(x, init_weights([3,3,384,512]), bias([512]), name="conv15")
     x = cnn_layer(x, init_weights([3,3,512,512]), bias([512]), name="conv16")
     x = tf.reshape(x,[-1, init_weights([1024,1024]).get_shape().as_list()[0]])
     x = tf.dropout(x,0.5)
     x = tf.add(tf.matmul(x, init_weights([1024,1024])), bias([1024]))
     x = tf.add(tf.matmul(x, init_weights([1024,1024])), bias([1024]))
     return x


X = tf.placeholder(tf.float32,[None,724,724,3],name = "Input_data")
Y = tf.palceholder(tf.float32,[None,5], name = "InputLabels")

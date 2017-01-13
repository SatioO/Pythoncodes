#train.py

from alexnet.helper import *
from alexnet.alexnet import *
import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.utils import shuffle
from tqdm import tnrange,tqdm_notebook


epochs = 50
batch_size = 4

logs_path = "tensorflow_logs/alexfish"
model_path  = "models/alex"

img_list = img_location_reader("data")
train,test = train_test_split(imglist,test_size=0.3,random_state=0)
train,valid = train_test_split(train,test_size=0.1,random_state=0)

x_dev = img_location_list(train,over_sample=True)
print (len(x_dev))

img_dummy_label = pd.get_dummies(list(image_label(x_dev).keys()))

x_val,y_val = val_test_image_reader(valid,size=(256,256),normalize = True)
# x_test,y_test = val_test_image_reader(test,size=(256,256))

print (x_val.shape)


X = tf.placeholder(tf.float32,[None,256,256,3],name = "Input_data")
Y = tf.placeholder(tf.float32,[None,8], name = "InputLabels")



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

pred = alexnet_model(X,weights,bias)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,Y))
#optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(0.001,decay=0.9,momentum=0.9,epsilon=1e-10).minimize(cost)

#Evaluate the model
correct_pred= tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#Initialize all the variables
init = tf.initialize_all_variables()

saver = tf.train.Saver()

# Create summary to monitor cost tensor , accuracy tensor
tf.scalar_summary("loss",cost)
tf.scalar_summary("accuracy",accuracy)

merged_summary_op = tf.merge_all_summaries()


with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(logs_path,graph=tf.get_default_graph())
    val_acc = []
    for ep in tqdm_notebook(range(epochs),desc="1st loop"):
        img_list = shuffle(x_dev,random_state=0)
        for i in tqdm_notebook(range(int(len(img_list)/batch_size)),desc="2nd loop",leave=False):
            img_loc = img_list[i*batch_size:(i+1)*batch_size]
            X_image, Y_image = dev_image_reader(img_loc,img_dummy_label,size = (256,256),normalize = True)
            _,c,summary = sess.run([optimizer,cost,merged_summary_op],feed_dict={X : np.float32(X_image), Y : np.float32(Y_image)})
            train_loss, train_acc = sess.run([cost,accuracy], feed_dict={X : np.float32(X_image), Y : np.float32(Y_image)})
            if i % 100 == 0:
                x = []
                for j in tqdm_notebook(range(len(x_val)),desc="3rd loop",leave=False):
                    xx_val = x_val[j][np.newaxis,:,:,:]
                    value = sess.run([tf.nn.softmax(pred)], feed_dict={X : np.float32(xx_val)})
                    x.append(value[0])
                x = np.concatenate(x)
                valid_acc = sess.run([tf.reduce_mean(tf.cast(tf.equal(tf.argmax(x,1),tf.argmax(Y,1)),tf.float32))],\
                        feed_dict={Y : np.float32(y_val)})
                print ("Train loss ="+"{:.6f}".format(train_loss),\
                       "Train acc ="+"{:.6f}".format(train_acc),\
                      "valid acc ="+"{:.6f}".format(valid_acc[0]))
                val_acc.append(valid_acc[0])
                if max(val_acc) == valid_acc[0]:
                    saver.save(sess,os.getcwd()+model_path+"_"+str(ep)+"_"+str(i))
        print ("epoch: ", ep)
    print("Optimization Finished!")

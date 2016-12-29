# train the model
from helper import *
import tensorflow as tf
import numpy as np
from model1 import *

folderlist = filelist("data")

X_dev,X_test = train_test(folderlist, test_size=0.3, random_state = 0)
X_dev, X_valid = train_test(X_dev,test_size=0.5,random_state=0)

print ("X_dev",sum([len(X_dev[i]) for i in X_dev.keys()]))
print ("X_valid",sum([len(X_valid[i]) for i in X_valid.keys()]))
print ("X_test",sum([len(X_test[i]) for i in X_test.keys()]))

logs_path = "/tmp/tensorflow_logs/example"
model_path = "/models/t1"

# Read the valid and test Images
x_valid_image,x_valid_label = Valid_Image_data_generator(X_dev,resize = (256,334),Transformation = False, scaling = True)
x_test_image,x_test_label = Valid_Image_data_generator(X_test,resize = (256,334),Transformation = False, scaling = True)


X = tf.placeholder(tf.float32,[None,256,334,3],name = "Input_data")
Y = tf.placeholder(tf.float32,[None,8], name = "InputLabels")

weights = {
"w1": init_weights([3,3,3,16]),
"w2": init_weights([2,2,16,16]),
"w3": init_weights([2,2,16,32]),
"w4": init_weights([2,2,32,32]),
"w5": init_weights([2,2,32,64]),
"w6": init_weights([2,2,64,64]),
"w7": init_weights([384,256]),
"w8": init_weights([256,64]),
"w9": init_weights([64,8])
}

bias = {
"b1":bias([16]),
"b2":bias([16]),
"b3":bias([32]),
"b4":bias([32]),
"b5":bias([64]),
"b6":bias([64]),
"b7":bias([256]),
"b8":bias([64]),
"b9":bias([8])
}

pred = model(X,weights["w1"],weights["w2"],weights["w3"],weights["w4"],weights["w5"],weights["w6"],
weights["w7"],weights["w8"],weights["w9"],bias["b1"],bias["b2"],bias["b3"],bias["b4"],bias["b5"],bias["b6"],bias["b7"],
bias["b8"],bias["b9"])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

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

training_iters = 24736
epochs = 10
batch_size = 4 
display_step = 1

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(logs_path,graph = tf.get_default_graph())
    step = 1
    # keep training until it reach max interations
    for ep in range(epochs):
        while step*batch_size < training_iters:
            (batch_x,batch_y) = next(Dev_Image_data_generator(folderlist,resize = (256,334),Transformation = True, scaling = True, batch_size = 4))
            _,c,summary = sess.run([optimizer,cost_merged_summary_op],feed_dict={X : np.float32(batch_x), Y : np.float32(batch_y)})
            if step % display_step == 0:
                train_loss, train_acc = sess.run([cost,accuracy], feed_dict={X : np.float32(batch_x), Y : np.float32(batch_y)})
            step += 1
        print ("epoch: ", ep)
        valid_loss, valid_acc = sess.run([cost,accuracy], feed_dict={X : np.float32(x_valid_image), Y : np.float32(x_valid_label)})
        print ("Train loss ="+"{:.6f}".format(train_loss),\
        "Train acc ="+"{:.6f}".format(train_acc),\
        "valid loss ="+"{:.6f}".format(valid_loss),\
        "valid acc ="+"{:.6f}".format(valid_acc))
    print("Optimization Finished!")
    test_loss, test_acc = sess.run([cost,accuracy], feed_dict={X : np.float32(x_test_image), Y : np.float32(x_test_label)})
    print ("test loss ="+"{:.6f}".format(test_loss),\
    "Test acc ="+"{:.6f}".format(test_acc))
    save_path = saver.save(sess,model_path)


# # Running  a new session
# print ("Starting 2nd session")
# with tf.Session() as sess:
#     sess.run(init)
#     load_path = saver.restore(sess,model_path)
#     # .... do what ever you feel like



# print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
#   "{:.6f}".format(loss) + ", Training Accuracy= " + \
#   "{:.5f}".format(acc))

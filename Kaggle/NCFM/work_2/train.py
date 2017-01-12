# train the model
from helper import *
import tensorflow as tf
import numpy as np
from model1 import *

batch_size = 8
logs_path = "/tmp/tensorflow_logs/example"
model_path = "/models/t1"

# split the data into train,valid and test
imglist = img_location_reader("data")
train,test = train_test_split(imglist,test_size=0.3,random_state=0)
train,valid = train_test_split(train,test_size=0.3,random_state=0)


# read the test and valid images



X = tf.placeholder(tf.float32,[None,256,256,3],name = "Input_data")
Y = tf.placeholder(tf.float32,[None,8], name = "InputLabels")

weights = {
"w1": init_weights([3,3,3,16]),
"w2": init_weights([2,2,16,16]),
"w3": init_weights([2,2,16,32]),
"w4": init_weights([2,2,32,32]),
"w5": init_weights([2,2,32,64]),
"w6": init_weights([2,2,64,64]),
"w7": init_weights([256,256]),
"w8": init_weights([256,64]),
"w9": init_weights([64,8])
}

bias = {
"b1":init_bias([16]),
"b2":init_bias([16]),
"b3":init_bias([32]),
"b4":init_bias([32]),
"b5":init_bias([64]),
"b6":init_bias([64]),
"b7":init_bias([256]),
"b8":init_bias([64]),
"b9":init_bias([8])
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


# splitting is done .
img_list = img_location_list(train,over_sample = True)
img_dummy_label = pd.get_dummies(image_label(img_list).keys())

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(logs_path,graph=tf.get_default_graph())
    for ep in tqdm(range(epochs)):
        img_list = shuffle(img_list,random_state=0)
        for i in range(len(img_list)/batch_size):
            img_loc = img_list[i*batch_size:(i+1)*batch_size]
            images = []
            images_label = []
            for j in range(len(img_loc)):
                x_image,y_label= image_read(img_loc[j])
                x_images = agument_data(x_image)
                images.append(x_images)
                images_label.append([img_dummy_label[y_label] for x in range(len(x_images))])
                X_image, Y_image = np.concatenate(images),np.concatenate(images_label)
            _,c,summary = sess.run([optimizer,cost,merged_summary_op],feed_dict={X : np.float32(X_image), Y : np.float32(Y_image)})
            train_loss, train_acc = sess.run([cost,accuracy], feed_dict={X : np.float32(batch_x), Y : np.float32(batch_y)})
            print ("epoch ="+"{:.6f}".format(ep),"step ="+"{:.6f}".format(step),\
               "Train loss ="+"{:.6f}".format(train_loss),\
               "Train acc ="+"{:.6f}".format(train_acc))
        print ("epoch: ", ep)
        valid_loss, valid_acc = sess.run([cost,accuracy], feed_dict={X : np.float32(x_valid_image), Y : np.float32(x_valid_label)})
        train_loss, train_acc = sess.run([cost,accuracy], feed_dict={X : np.float32(batch_x), Y : np.float32(batch_y)})
        print ("Train loss ="+"{:.6f}".format(train_loss),\
            "Train acc ="+"{:.6f}".format(train_acc),\
            "valid loss ="+"{:.6f}".format(valid_loss),\
            "valid acc ="+"{:.6f}".format(valid_acc))
    print("Optimization Finished!")
    test_loss, test_acc = sess.run([cost,accuracy], feed_dict={X : np.float32(x_test_image), Y : np.float32(x_test_label)})
    print ("test loss ="+"{:.6f}".format(test_loss),\
        "Test acc ="+"{:.6f}".format(test_acc))
        save_path = saver.save(sess,model_path)

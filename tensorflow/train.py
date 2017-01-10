# train the model
from helper import *
import tensorflow as tf
import numpy as np
from model1 import *
from sklearn.cross_validation import train_test_split

batch_size = 8
logs_path = "/tmp/tensorflow_logs/example"
model_path = "/models/t1"

# split the data into train,valid and test
labels = pd.read_csv("/Users/Satish/Downloads/DR/trainLabels.csv",index_col=["image"])

file_loc = img_location_reader()
train,test = train_test_split(file_loc,test_size=0.1,random_state=0)
train,valid = train_test_split(train,test_size=0.1,random_state=0)


# read the test and valid images



X = tf.placeholder(tf.float32,[None,724,724,3],name = "Input_data")
Y = tf.placeholder(tf.float32,[None,5], name = "InputLabels")

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

weights = {
"w1": init_weights([3,3,3,16]),
"w2": init_weights([3,3,16,16]),
"w3": init_weights([3,3,16,32]),
"w4": init_weights([3,3,32,32]),
"w5": init_weights([3,3,32,64]),
"w6": init_weights([3,3,64,64]),
"w7": init_weights([3,3,64,128]),
"w8": init_weights([3,3,128,128]),
"w9": init_weights([3,3,128,192]),
"w10": init_weights([3,3,192,192]),
"w11": init_weights([3,3,192,256]),
"w12": init_weights([3,3,256,256]),
"w13": init_weights([3,3,256,384]),
"w14": init_weights([3,3,384,384]),
"w15": init_weights([3,3,384,512]),
"w16": init_weights([3,3,512,512]),
"w17": init_weights([512,1024]),
"w18": init_weights([1024,1024]),
"w19": init_weights([1024,5])
}

bias = {
"b1":init_bias([16]),
"b2":init_bias([16]),
"b3":init_bias([32]),
"b4":init_bias([32]),
"b5":init_bias([64]),
"b6":init_bias([64]),
"b7":init_bias([128]),
"b8":init_bias([128]),
"b9":init_bias([192]),
"b10":init_bias([192]),
"b11":init_bias([256]),
"b12":init_bias([256]),
"b13":init_bias([384]),
"b14":init_bias([384]),
"b15":init_bias([512]),
"b16":init_bias([512]),
"b17":init_bias([1024]),
"b18":init_bias([1024])
}

pred = model(X,weights["w1"],weights["w2"],weights["w3"],weights["w4"],weights["w5"],weights["w6"],
weights["w7"],weights["w8"],weights["w9"],weights["w10"],weights["w11"],weights["w12"],weights["w13"],weights["w14"],weights["w15"],weights["w16"],weights["w17"],weights["w18"],weights["w19"],bias["b1"],bias["b2"],bias["b3"],bias["b4"],bias["b5"],bias["b6"],bias["b7"],
bias["b8"],bias["b9"],bias["b10"],bias["b11"],bias["b12"],bias["b13"],bias["b14"],bias["b15"],bias["b16"],bias["b17"],bias["b18"])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)

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
img_dummy_label = pd.get_dummies(labels["level"].unique())

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
                x_images = np.concatenate([contrast_channel_wise(x_images[i])[np.newaxis,:,:,:] for i in range(len(x_images))])
        		x_images = np.concatenate([resize(x_images[i],size=(724,724))[np.newaxis,:,:,:] for i in range(len(x_images))])
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

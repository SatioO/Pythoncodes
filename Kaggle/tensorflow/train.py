# train the model

logs_path = "/tmp/tensorflow_logs/example"
model_path = "/models/t1"


X = tf.placeholder(tf.float32,[None,256,334,3],name = "Input_data")
Y = tf.palceholder(tf.float32,[None,8], name = "InputLabels")

pred = model(X)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

#Evaluate the model
correct_pred= tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#Initialize all the variables
init = tf.initialize_all_variables()

saver = tf.train.Saver()

# Create summary to monitor cost tensor , accuracy tensor
tf.scalar.summary("loss",cost)
tf.scalar.summary("accuracy",accuracy)

merged_summary_op = tf.merge_all_summaries()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(logs_path,graph = tf.get_default_graph())
    step = 1
    # keep training until it reach max interations
    while step*batch_size < training_iters:
        (batch_x,batch_y) = Image_data_generator(folderlist,labels,resize = (256,334),Transformation = True, scaling = True, batch_size = 4)
        _,c,summary = sess.run([optimizer,cost_merged_summary_op],feed_dict={x : batch_x, y : batch_y})
        if step % display_step == 0:
            loss, acc = sess.run([cost,accuracy], feed_dict={x: batch_x, y: batch_y,
            keep_prob : 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x:x_test,
    y : y_test,
    keep_prob: 1.}))
    save_path = saver.save(sess,model_path)


# # Running  a new session
# print ("Starting 2nd session")
# with tf.Session() as sess:
#     sess.run(init)
#     load_path = saver.restore(sess,model_path)
#     # .... do what ever you feel like

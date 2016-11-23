exec(compile(open("lib.py","rb").read(),"lib.py","exec"))
exec(compile(open("pre_process.py","rb").read(),"pre_process.py","exec")) # see the file before you run
exec(compile(open("architecture.py","rb").read(),"architecture.py","exec"))
"""

All the Image files location will be in filelist 
- For each epoch we will
       - shuffle the entire list 
       - Pick a batch of first 128 Images 
       - Load the 128 Images 
       - Using filenames of these 128 Images, we will extract the labels also from lables folder
       - Dummify the labels [if doing classification]
       - Pass the data through neural net 
       - Compute the error 
       - Back propogate again

"""

"""
Training on batch

"""

X_train,X_val,train_Y, val_Y = train_test_split(zip(left_eye,right_eye),lr_label,
	test_size = 0.10, random_state = 20)

le_val = np.array([io.imread(X_val[i][0]) for i in np.arange(len(X_val))])
re_val = np.array([io.imread(X_val[i][1]) for i in np.arange(len(X_val))])
val_Y = np.array(val_Y)

epochs = 10 
batch_size = 128 

## Training the neural network 

model_train = model(input_shape)


for i in range(epochs):
	np.random.shuffle(zip(X_train,train_Y))
	n_batches = len(X_train)//batch_size
	for batch_i in range(n_batches):
		X = X_train[batch_i * batch_size: (batch_i + 1) * batch_size] # select the batch of Images
		leX = np.array([io.imread(X[i][0]) for i in np.arange(len(X))])
		reX = np.array([io.imread(X[i][1]) for i in np.arange(len(X))])
		out = np.array(train_Y[batch_i * batch_size: (batch_i + 1) * batch_size])
		loss = model_train.train_on_batch([leX,reX],out)
		test_loss = model_train.test_on_batch([le_val,re_val],val_Y)
		print (i,batch_i,loss,test_loss)
	score = model_train.evaluate([le_val,re_val],val_Y,verbose=0)
	print ("Test Score",score)


## Define the architecture 

"""
- Get a batch of 128 left Images 
- from the same get a batch of 128 right Images 
- run through different architectures 
- merge them 

"""

""" will be using tensorflow backend """


"""
Some useful links 
- https://github.com/fchollet/keras/issues/148



"""

input_shape = (64,64,3) ## changes if the backend is theano to (3,64,64)
def model(imput_shape):
	# Left Eye
	model1 = Sequential()
	model1.add(Convolution2D(10,3,3,border_mode="valid",input_shape=input_shape))
	model1.add(Activation("relu"))
	model1.add(Convolution2D(20,3,3,border_mode="valid"))
	model1.add(Activation("relu"))
	model1.add(MaxPooling2D(pool_size  = (2,2)))
	model1.add(Convolution2D(30,3,3,border_mode="valid"))
	model1.add(Activation("relu"))
	model1.add(MaxPooling2D(pool_size  = (2,2)))
	model1.add(Convolution2D(40,3,3,border_mode="valid"))
	model1.add(Activation("relu"))
	model1.add(MaxPooling2D(pool_size  = (2,2)))

	# Right Eye 
	model2 = Sequential()
	model2.add(Convolution2D(10,3,3,border_mode="valid",input_shape=input_shape))
	model2.add(Activation("relu"))
	model2.add(Convolution2D(20,3,3,border_mode="valid"))
	model2.add(Activation("relu"))
	model2.add(MaxPooling2D(pool_size  = (2,2)))
	model2.add(Convolution2D(30,3,3,border_mode="valid"))
	model2.add(Activation("relu"))
	model2.add(MaxPooling2D(pool_size  = (2,2)))
	model2.add(Convolution2D(40,3,3,border_mode="valid"))
	model2.add(Activation("relu"))
	model2.add(MaxPooling2D(pool_size  = (2,2)))

	merged_model = Sequential()
	merged_model.add(Merge([model1,model2],mode= "concat",concat_axis = 1))
	merged_model.add(Flatten())
	merged_model.add(Dense(1024))
	merged_model.add(Activation("relu"))
	merged_model.add(Dropout(0.3))
	merged_model.add(Dense(108))
	merged_model.add(Activation("relu"))
	merged_model.add(Dropout(0.3))
	merged_model.add(Dense(1)) # this can be changed to linear regression if needed 
	merged_model.add(Activation("softmax")) 
	return merged_model 







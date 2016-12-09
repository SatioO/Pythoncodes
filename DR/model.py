"""

Keras - Model Building 
Architecture - 
    32C5 - 64C3 - 96C3 - 
    128C3 - 160C3 - 190C3 - 
    224C3 - 256C3 - 228C3 - 
    320C2 - 352C1 - 5N


 Input - 270*270

"""

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten 
from keras.layers import Convolution2D,MaxPooling2D
from keras.optimizers import SGD 

""" this works using max-pooling instead of fractional max pooling """


from keras.

input_shape = (270,270,3)

def model(input_shape):
	model = Sequential()
	model.add(Convolution2D(32,5,5,border_mode="valid",input_shape=input_shape))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.10))
	model.add(Convolution2D(64,3,3,border_mode="valid"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.10))
	model.add(Convolution2D(96,3,3,border_mode="valid"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.20))
	model.add(Convolution2D(128,3,3,border_mode="valid"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.20))
	model.add(Convolution2D(160,3,3,border_mode="valid"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.30))
	model.add(Convolution2D(190,3,3,border_mode="valid"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.40))
	model.add(Convolution2D(256,2,2,border_mode="valid"))
	model.add(Activation("relu"))
	model.add(Dropout(0.50))
	model.add(Convolution2D(320,1,1,border_mode="valid"))
	model.add(Flatten())
	model.add(Dense(5,init="uniform"))
	model.add(Activation("softmax"))
	sgd = SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
	model.compile(loss = "categorical_crossentropy",optimizer=sgd,metrics =["accuracy"])
	return model 

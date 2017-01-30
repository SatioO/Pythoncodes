"""
| --- output
|     | ---- accuracy
|     | ---- models
| --- pyimagesearch
|     | ---- __init__.py
|     | ---- cnn
|     |      | ---- __init__.py
|     |      | ---- convenetfactory.py
| --- test_images
| --- train_network.py
| --- test_network.py
"""

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.models import Sequential

class ConvnetFactory:
    def __init__(self):
        pass

    def build(name, *args, **kargs):
        #define the network (i.e., string => function) mappings
        mappings = {
        "shallownet": ConvnetFactory.ShallowNet,
        "lenet":ConvnetFactory.LeNet,
        "karpathynet": ConvnetFactory.KarpathyNet,
        "minivggnet": ConvnetFactory.MiniVGGNet
        }
        #grab the builder function from the mappings dictionary
        builder = mappings.get(name, None)

        # if the builder is None, then there is not a function that can be used to
        # build to the network , so return None
        if builder is None:
            return None

        # otherwise, build the network architecture
        return builder(*args, **kargs)

    def ShallowNet(numChannels, imgRows, imgCols, numClasses, **kwargs):
        model = Sequential()

        # define the first (and only) CONV => RELU Layer
        model.add(Convolution2D(32, 3, 3, border_mode="same",
        input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))

        # add an FC layer followed by the softmax classifier
        model.add(Flatten())
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        return model

    def LeNet(numChannels, imgRows, imgCols, numClasses, activation="tanh", **kwargs):
        model = Sequential()

        # define the first (and only) CONV => RELU Layer
        model.add(Convolution2D(20, 5, 5, border_mode="same",
        input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Convolution2D(50,5,5, border_mode="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        #define the second FC layer
        model.add(Dense(numChannels))
        return model

    def KarpathyNet(numChannels, imgRows, imgCols, numClasses, dropout = False, **kwargs):
        model = Sequential()

        model.add(Convolution2D(16, 5, 5, border_mode="same", input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

        if dropout:
            model.add(Dropout(0.25))

        model.add(Convolution2D(20, 5, 5, border_mode="same", input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

        if dropout:
            model.add(Dropout(0.25))

        model.add(Convolution2D(20, 5, 5, border_mode="same", input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

        if dropout:
            model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        return model

    def MiniVGGNet(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
        model = Sequential()

        model.add(Convolution2D(32,3,3,border_mode="same", input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))
        model.add(Convolution2D(32,3,3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

        if dropout:
            model.add(Dropout(0.25))

        model.add(Convolution2D(64, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

        if dropout:
            model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))

        if dropout:
            model.add(Dropout(0.5))

        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        return model

    

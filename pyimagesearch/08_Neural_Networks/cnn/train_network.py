from pyimagesearch.cnn import ConvnetFactory
from keras.optimizers import SGD
fro keras.datasets import cifar10
from keras.utils import np_utils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--network", required = True, help = "name of the network to build")
ap.add_argument("-m" "--model", required = True, help = "path to output model file")
ap.add_argument("-d", "--dropout", type = int, default=-1, help="weather or not dropout should be used")
ap.add_argument("-f", "--activation", type="str", default="tanh", help = "Activation function to use (LeNet only)")
ap.add_argument("-e", "--epochs", type = int, default = 20, help = "# of epochs")
ap.add_argument("-b", "--batch-size", type = int, default = 32, help="size of mini-batches passed to network")
ap.add_argument("-v", "--verbose", type = int, default=1, help ="verbosity level")
args = vars(ap.parse_args())

((trainData, trainLabels), (testData, testLabels)) = cifar10.load_data()
trainData = trainData.astype("float")/255.0
testData = testData.astype("float")/255.0

trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

kargs = {"dropout": args["dropout"]>0, "activation":args["activation"]}

#train the model using SGD
print ("[INFO] compiling model...")
model = ConvnetFactory.build(args["network"], 3, 32, 32, 10, **kargs)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov= True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# start the training process
print ("[INFO] starting training")
model.fit(trainData, trainLabels, batch_size = args["batch_size"], nb_epoch=args["epochs"],
verbose = args["verbose"])

#show the accuaracy on the testing set
(loss, accuaracy) = model.evaluate(testData, testLabels,
batch_size = args["batch_size"], verbose=args["verbose"])

print ("[INFO] accuracy:{:.2f}%".format(accuracy*100))

#dump the network architecture and weights to file
print ("[Info] saving the model")
model.save(args["model"])

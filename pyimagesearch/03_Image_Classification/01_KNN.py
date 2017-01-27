#- KNN
from sklearn.cross_validation import train_test_split
from sklearn.neighbours import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2

#load the MNIST digits dataset
mnist = datasets.load_digits()

# take the MNIST data and construct the training and testing split, using 75% of the
# data for training and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
	mnist.target, test_size=0.25, random_state=42)

# now, let's take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
	test_size=0.1, random_state=84)

# show the sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

#initialize the values of k for our KNeighborsClassifier along with the list of
#accuaracies for each value of K
kVals = range(1, 30, 2)
accuaracies = []

# loop over various values of 'k' for K- Nearest Neighbour classifier
for k in xrange(1, 30, 2):
    #training the k-Nearest Neighbour classifier with the current value of 'k'
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(trainData, trainLabels)

    #evaluate the model and update the accuaries list
    score = model.score(valData, valLabels)
    print ("k=%d, accuracy=%.2f%%"% (k, score*100))
    accuaracies.append(score)

# find the value of k that has the largest accuracy
i = np.argmax(accuaracies)
print ("k%d achieved highest accuaracy of %.2f%% on validation data" % (kVals[i], accuaracies[i]*100))

# re-train our classifier using the best k value and predict the labels of the test data
model = KNeighborsClassifier(n_neighbors = kVals[i])
model.fit(trainData, trainLabels)
predictions = model.predict(testData)

# show a final classification_report demonstrating the accuaracy of the classifier for each of the digits
print ("Evaluation on testing data")
print (classification_report(testLabels, predictions))

# Examining some of the individual predictions from our k-NN classifier
for i in np.random.randint(0, high= len(testLabels), size=(5,)):
    #grab the image and classify it
    image = testData[i]
    prediction = model.predict(image.reshape(1,-1))[0]

    # convert the image for a 64-dim array to an 8*8 image compatible with OpenCv
    # then resize it to 32*32 pixels so we can see it better
    image = image.reshape((8, 8)).astype("uint8")
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

    print("I think the digit is :{}".format(prediction))
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    

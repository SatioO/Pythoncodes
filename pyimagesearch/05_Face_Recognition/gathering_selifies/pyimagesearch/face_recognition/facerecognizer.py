#import the necessary packages
from collections import namedtuple
import cPickle #changed in python 3
import cv2
import os


#Define the face recognizer instance
FaceRecognizerInstance = namedtuple("FaceRecognizerInstance", ["trained", "labels"])

class FaceRecognizer:
    def __init__(self, recognizer, trained = False, labels = None):
        self.recognizer = recognizer
        self.trained = trained
        self.labels = labels

    def setLabels(self, labels):
        self.labels = labels

    def setConfidenceThreshold(self, confidenceThreshold):
        self.recognizer.setDouble("threshold", confidenceThreshold)

    def train(self,data, labels):
        if not self.trained:
            self.recognizer.train(data, labels)
            self.trained = True
            return

        #otherwise update the model
        self.recognizer.update(data, labels)

    def predict(self, face):
        (prediction, confidence) = self.recognizer.predict(face)

        if prediction == -1:
            return ("unknown",0)

        # return a tuple of the face label and confidence
        return (self.labels[prediction], confidence)

    def save(self, basePath):
        fri = FaceRecognizerInstance(trained=self.trained, labels=self.labels)

        if not os.path.exists(basePath+"/classifier.model"):
            f = open(basePath+"/classifier.model","w")
            f.close()

        # write the actual recognizer along with parameters to file
        self.recognizer.save(basePath + "/classifier.model")
        f = open(basePath + "/fr.cpickle", "w")
        f.close()

    def load(basePath):
        fri = cPickle.loads(open(basePath+"/fr.cpickle").read())
        recognizer = cv2.createLBPHFaceRecognizer()
        recognizer.load(basePath+"/classifier.model")

        return FaceRecognizer(recognizer, trained=fri.trained , labels=fri.labels)

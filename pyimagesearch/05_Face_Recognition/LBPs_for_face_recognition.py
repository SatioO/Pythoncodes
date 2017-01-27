""" IMPLEMENTING LBPs FOR FACE RECOGNITION """
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import argparse, imutils, cv2

ap =argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to CALTECH Faces dataset")
ap.add_argument("-s", "--sample-size", type = int , default = 10, help = "# of example samples")
args = vars(ap.parse_args())

(training, testing, names) = load_caltech_faces(args["dataset"], min_faces=21, test_size0.25)

le=LabelEncoder()
le.fit_transform(training.target)

print ("[INFO] training face detector")
recognizer = cv2.createLBPHFaceRecognizer(radius=2, neighbors=16, grid_x = 8, grid_y = 8)

# initialize the list of predictions and confidence scores
predictions=[]
confidence=[]

for i in range(len(testing.data)):
    (prediction,conf)=recognizer.predict(testing.data[i])
    predictions.append(prediction)
    confidence.append(conf)

print (classification_report(le.transform(testing.target), predictions, target_names = names))

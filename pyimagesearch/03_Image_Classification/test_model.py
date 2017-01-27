from sklearn.metrics import classification_report
from ise.descriptors.detectanddescribe import DetectAndDescibe
from ise.descriptors.rootshift import RootSHIFT
from ise.descriptors.pbow import PBOW
from ise.ir.bagofvisualwords import BagOfVisualWords
from imutils import paths
import numpy as np
import argparse
import cPickle
import imutils
import cv2 , glob, os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help = "Path to input images directory")
ap.add_argument("-c", "--codebook", required=True, help = "Path to the codebook")
ap.add_argument("-l", "--levels", type = int, default =2, help = "# of pyramid levels to generate")
ap.add_argument("-m", "--model", required=True, help = "Path to the classifier")
args = vars(ap.parse_args())

# initialize the keypoint detector, local invariant descriptor, and the descriptor
# pipeline
detector = cv2.FeatureDetector_create("GFTT")
descriptor = RootSIFT()
dad = DetectAndDescribe(detector, descriptor)

#load the codebook vocabulary and initialize the bag-of-visual-words transformer
vocab = cPickle.loads(open(args["codebook"]).read())
bovw = BagOfVisualWords(vocab)
pbow = PBOW(bovw, numLevels = args["levels"])

#load the classifier and grab the list of image paths
model = cPickle.loads(open(args["model"]).read())
imagePaths = list(paths.list_images(args["images"]))

trueLabels = []
predictedLabels = []

for imagePath in imagePaths:
    trueLabels.append(imagePath.split("/")[-2])
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(320, image.shape[1]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # describe the image
    (kps, descs) = dad.describe(gray)
    hist = pbow.describe(gray.shape[1], gray.shape[0], kps, descs)
    prediction = model.predict(hist)[0]
    predictedLabels.append(prediction)

print (classification_report(trueLabels,predictedLabels))

for i in np.random.choice(np.arange(0,len(imagePaths)), size=(20,), replace=False):
    (labels,filename)=featuresDB["image_ids"][i].split(":")
    image =cv2.imread(imagePaths[i])

    #show the prediction
    filename = imagePaths[i][imagePaths[i].rfind("/") + 1:]
    print ("[PREDICTION] {}:{}".format(filename, predictedLabels[i]))
    cv2.putText(image, predictedLabels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0,255,0),2)
    cv2.imshow("Image",image)
    cv2.waitKey(0)


from histograms.descriptors.labhistogram import LabHistogram
from sklearn.cluster import kMeans
from imutils import path
import numpy as np
import argparse
import cv2, glob

# compress the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help = "path to the input dataset directory")
ap.add_argument("-k", "--clusters", type = int , default= 2, help="# of clusters to generate")
args = vars(ap.parse_args())


desc = LabHistogram([8, 8, 8])
data = []

# grab the image path from the dataset directory
imagePaths = glob.glob("")
imagePaths = np.array(sorted(imagePaths))

# loop over the input dataset of images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    hist = desc.describe(image)
    data.append(hist)

# cluster the color histograms
clt = KMeans(n_clusters= args["clusters"])
labels = clt.fit_predict(data)

#loop over the unique labels
for label in np.unique(labels):
    # grab all the image paths that are assigned to the current label
    labelPaths = imagePaths[np.where(labels == label)]

    #loop over the image paths that belong to the current label
    for (i, path) in enumerate(labelPaths):
        image = cv2.imread(path)
        cv2.imshow("Cluster {}, image#{}".format(label+1,i+1),image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

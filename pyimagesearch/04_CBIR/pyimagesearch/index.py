# import the necessary packages
from pyimagesearch.cbir import HSVDescriptor
from imutils import paths
import argparse
import cv2
import glob
import itertools

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True, help = "Path to where the features index will be stored")
args = vars(ap.parse_args())

#initialize the color descriptor and open the output index  file for writing
desc = HSVDescriptor((4, 6, 3))
output = open(args["index"], "w")

imagePaths = glob.glob("/Users/Satish/Downloads/kaggle-fish/*")
imagePaths = [glob.glob(imagePaths[i]+"/*") for i in range(len(imagePaths))]
imagePaths = list(itertools.chain(*imagePaths))


for (i, imagePath)  in enumerate(imagePaths):
    filename = imagePath.rsplit("/")[-1]
    image = cv2.imread(imagePath)

    # describe the image
    features = [str(x) for x in features]
    output.write("{}.{}\n".format(filename,",".join(features)))

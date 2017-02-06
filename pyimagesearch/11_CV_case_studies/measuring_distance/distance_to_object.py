#import the necessary packages
import pyimagesearch.markers import DistanceFinder
from imutils import paths
import argparse
import cv2


# construct the argument parser and parse the command line argument
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reference", required = True, help = "path to the reference image")
ap.add_argument("-w","--ref-width-inches", required = True, help = "reference object width in inches")
ap.add_argument("-d", "--ref-distance-inches", required = True, help = "distance to reference object in inches")
ap.add_argument("-i", "--images", required = True, help="path to the directory containing images to test")
args = vars(ap.parse_args())

#load the reference image and resize it
refImage = cv2.imread(args["reference"])
refImage = imutils.resize(refImage, height = 700)

# initialize the distance finder
df = DistanceFinder(args["ref-width-inches"], args["ref-distance-inches"])
refMarker = DistanceFinder.findSquareMarker(refImage)

#visualize the results on the reference image and display it
refImage = df.draw(refImage, refMarker, df.distance(refMarker[2]))
cv2.imshow("reference", refImage)


for i in paths.list_images(args["images"]):
    filename=i[i.rfind("/")+1:]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, height = 700)
    print ("[INFO] processing {}".formate(filename))

    marker = DistanceFinder.findSquareMarker(image)

    if marker is None:
        print ("[INFO] could not find marker for {}".format(filename))
        continue

    distance = df.distance(marker[2])

    #visualize the result on the image and display it
    image = df.draw(image, marker, distance)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


# python distance_to_object.py --reference imagelocation --ref-width-inches 4.0 --ref-distance-inches 24.0 --images images

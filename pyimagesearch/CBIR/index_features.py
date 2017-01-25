from ise.descriptors.detectanddescribe import DetectAndDescribe
from ise.descriptors.rootsift import RootSIFT
from pyimagesearch.indexer import FeatureIndexer
from imutils import paths
import argparse, imutils, cv2, glob, os


ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required = True, help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-f","--features_db",required = True, help = "Path to where feature database will be stored")
ap.add_argument("-d","--approx_images",type=int,default=500, help = "Approximate # of images in the dataset")
ap.add_argument("-d","--max_buffer_size",type = int, default = 5000, help = "Maximum buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())

#initialize the keypoint detector , local invariant descriptor , and the descriptor pipeline
detector = cv2.xfeatures2d.SIFT_create()


#initialize the feature indexer
fi = FeatureIndexer(args["feature_db"],estNumImages=args["approx_images"], maxBufferSize = args["max_buffer_size"])



for (i, imagePath) in enumerate(glob.glob("")):
    # check to see if progress should be displayed
    if i > 0 and i % 10 = 0:
        fi._debug("processed {} images".format(i), msgType="[PROGRESS]")

        #Extract the filename from the image path, then load the image itself
        filename = imagePath.rsplit("/")[-1]
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width = 320)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #describe the image
        (kps, decs) = detector.describe(image)

        # if either the keypoints or descriptors is None, then ignore the image
        if kps is None or descs is None:
            continue

        # index the features
        fi.add(filename, kps, descs)

#finish the indexing process
fi.finish()

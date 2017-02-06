"""
- 1. Extract and index features.
- 2. Cluster features to form a visual vocabulary.
- 3. Quantize the feature vectors to form a BOVW histogram for each image in the dataset.
- 4. Train the classifier on top of the histogram representations.

"""

from ise.descriptors.detectanddescribe import DetectAndDescibe
from ise.descriptors.rootshift import RootSHIFT
from ise.indexer.featureindexer import FeatureIndexer
from imutils import paths
import argparse, imutils, random, cv2, glob

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help = "Path to dataset")
ap.add_argument("-f", "--features-db", required=True, help = "Path to where the features data will be stored")
ap.add_argument("-a", "--approx-images", type = int, default=500, help="Approx # of images in the dataset")
ap.add_argument("-b", "--max-buffer-size", type=int, default=500, help = "Maximum buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())

detector = cv2.xfeatures2d.SIFT_create()
descriptor = RootSIFT()
dad = DetectAndDescribe(detector, descriptor)

fi = FeatureIndexer(args["features_db"],estNumImages=args["approx_images"], maxBufferSize = args["max_buffer_size"], verbose = True)
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

for (i, imagePath) in enumerate(imagePaths):
    if i>0 and i%10 ==0:
        bi._debug("processed {} images".format(i), msgType="[PROGRESS]")

    p = imagePath.split("/")
    imageID = "{}:{}".format(p[-2],p[-1])

    #load the image and prepare it for description
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(320,image.shape[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #describe the image
    (kps, descs) = dad.describe(image)
    if kps is None or descs is None:
        continue
    fi.add(imageID, kps, descs)

fi.finish()

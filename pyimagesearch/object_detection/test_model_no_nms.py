from pyimagesearch.object_detection.objectdetector import ObjectDetector
from pyimagesearch.descriptors.hog import HOG
from pyimagesearch.utils.conf import Conf
import imutils
import argparse
import _pickle as cPickle
import cv2

# construct the argument parser and parse the argument
ap.add_argument("-c", "--conf", required = True, help = "path to configuration file")
ap.add_argument("-i", "--image", required = True, help = "path to the image being classified")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

#load the classifier
model = cPickle.loads(open(conf["classifier_path"]).read())
hog = HOG(orientations=conf["orientations"], pixelsPerCell = tuple(conf["pixels_per_cell"]),
cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
od = ObjectDetector(model, hog)

# loads the image and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=min(260,image.shape[1]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detect the objects
(boxes, prob) = od.detect(gray, conf["window_dim"], winStep=conf["window_step"],
pyramidScale=conf["pyramid_scale"], minProb=conf["min_probability"])

#loop over the bounding boxes and draw then
for (startX, startY, endX, endY) in boxes:
    cv2.rectangle(image,(startX, startY),(endX,endY), (0,0,255), 2)

cv2.imshow("Image",image)
cv2.waitKey(0)

from pyimagesearch.object_detection.objectdetector import ObjectDetector
from pyimagesearch.descriptors.hog import HOG
from pyimagesearch.utils.dataset import dataset
from pyimagesearch.utils.conf import Conf
from imutils import paths
import numpy as np
import argparse
import _pickle as cPickle
import cv2
import random

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required = True, help = "path to configuration file")
args = vars(ap.parse_args())

conf = Conf(args["conf"])
data = []

# load the classifier, then initialize the HOG descriptor
with open(r"model.pickle", "rb") as input_file:
    model = cPickle.load(input_file)
hog = HOG(orientations=conf["orientations"], pixelsPerCell = tuple(conf["pixels_per_cell"]),
cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
od = ObjectDetector(model, hog)

dstPaths = glob.glob(conf["image_distractions"]+"/*")
dstPaths = [glob.glob(dstPaths[i]+"/*") for i in range(len(dstPaths))]
dstPaths = list(itertools.chain(*dstPaths))
dstPaths = random.sample(dstPaths, conf["hh_num_distraction_images"])


# loop over the distraction images
for (i, imagePath) in enumerate(dstPaths):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (boxes, probs) = od.detect(gray, conf["window_dim"], winStep=conf["window_step"],
    pyramidScale=conf["hh_pyramid_scale"], minProb=conf["hn_min_probability"])

    for (prob, (startX, startY, endX, endY)) in zip(probs, boxes):
        roi = cv2.resize(gray[startX:endY, startX:endX], tuple(conf["window_dim"]),
        interpolation = cv2.INTER_AREA)
        features = hog.describe(roi)
        data.append(np.hstack([[prob], features]))


data = np.array(data)
data = data[data[:0].argsort()[::-1]]

dataset.dump_dataset(data[:,1:],[-1]*len(data), conf["features_path"], "hard_negatives", writeMethod = "a")

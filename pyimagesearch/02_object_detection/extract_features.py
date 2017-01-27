from sklearn.feature_extraction.image import extract_patches_2d
from pyimagesearch.object_detection import helpers
from pyimagesearch.descriptors.hog import HOG
from pyimagesearch.utils import dataset
from pyimagesearch.utils.conf import Conf
from imutils import paths
from scipy import io
import numpy as np
from tqdm import tqdm
import argparse
import random
import cv2
import itertools
import glob
import json
from scipy.io import loadmat


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required = True, help = "path to the configuration file")
args = vars(ap.parse_args())


conf = Conf(args["conf"])

#initialize the HOG descriptors along with the list of data and labels
hog = HOG(orientations=conf["orientations"], pixelsPerCell = tuple(conf["pixels_per_cell"]),
cellsPerBlock= tuple(conf["cells_per_block"]), normalize=conf["normalize"])

data = []
labels = []

# grad the set of ground truth images and select a percentage of them for training
trnPaths = glob.glob(conf["image_dataset"]+"/*")
trnPaths = random.sample(trnPaths, int(len(trnPaths)* conf["percent_gt_images"]))
print ("describing training ROI")


# loop over the training paths
for (i, trnPath) in tqdm(enumerate(trnPaths)):
    image = cv2.imread(trnPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageID = trnPath.rsplit("/")[-1].rsplit("_")[1].replace(".jpg","")
    p = conf["image_annotations"]+"/annotation_{}.mat".format(imageID)
    bb = loadmat(p)["box_coord"][0]
    roi = helpers.crop_ct101_bb(image, bb, padding=conf["offset"], dstSize=tuple(conf["window_dim"]))
    rois = (roi, cv2.flip(roi, 1)) if conf["use_flip"] else (roi,)

    for roi in rois:
        features = hog.describe(roi)
        data.append(features)
        labels.append(1)

# grad the set of ground truth images and select a percentage of them for training
dstPaths = glob.glob(conf["image_distractions"]+"/*")
dstPaths = [glob.glob(dstPaths[i]+"/*") for i in range(len(dstPaths))]
dstPaths = list(itertools.chain(*dstPaths))

for i in tqdm(range(conf["num_distraction_images"])):
    image = cv2.imread(random.choice(dstPaths))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patches = extract_patches_2d(image, tuple(conf["window_dim"]), max_patches=conf["num_distraction_per_image"])

    for patch in patches:
        features = hog.describe(patch)
        data.append(features)
        labels.append(-1)

print ("dumping features and labels to file")
dataset.dump_dataset(data, labels, conf["features_path"],"features")

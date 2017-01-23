""" train_model.py """

from pyimagesearch.utils import dataset
from pyimagesearch.utils.conf import Conf
from sklearn.svm import SVC
import numpy as np
import argparse
import _pickle as cPickle

ap = argparse.ArgumentParser()
ap.add_argument("-c","--conf", required = True, help = "path to configuration file")
ap.add_argument("--hard_negatives", type =int, default = -1,help = "flag indicating weather or not hard negatives should be used")
args = vars(ap.parse_args())


conf = Conf["conf"]
(data, labels) = data.load_dataset(conf["features_path"], "features")

# check if the hard-negatives flag was supplied
if args["hard_negatives"]>0:
    (hardData, hardLabels) = dataset.load_dataset(conf["features_path"], "hard_negatives")
    data = np.vstack([data, hardData])
    labels = np.hstack([labels, hardLabels])


print ("[Info] training classifier")
model = SVC(kernel="linear", C = conf["C"], probability = True, random_state=42)
model.fit(data, labels)

#"dump the classifier file"
print ("[INFO] dump classifier")
with open(r"model.pickle", "wb") as output_file:
    cPickle.dump(model, output_file)

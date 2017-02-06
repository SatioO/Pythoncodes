
from pyimagesearch.utils.dataset import build_cifar10
from argparse
import glob

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to the input dataset")
ap.add_argument("-o", "--output", required = True, help = "path to the output directory")
ap.add_argument("-t", "--train", required = True, help = "path to the output training file")
ap.add_argument("-v", "--test", required = True, help = "path to the output testing file")
args = vars(ap.parse_args())

# open the training file for writing and building the set of images
print ("[INFO] gathering training data ..")
f = open(args["train"], "w")
build_cifar10(glob.glob("{}/data_batch_*".format(args["dataset"])), args["output"], f)
f.close()


# open the training file for writing and building the set of images
print ("[INFO] gathering testing data ..")
f = open(args["test"], "w")
build_cifar10(glob.glob("{}/data_batch_*".format(args["dataset"])), args["output"], f)
f.close()

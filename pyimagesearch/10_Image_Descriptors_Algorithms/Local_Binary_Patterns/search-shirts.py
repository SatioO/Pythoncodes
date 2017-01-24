from Local_Binary_Patterns.descriptors.localbinarypatterns import *
from imutils import paths
import numpy as np
import argparse, cv2, glob, os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to the dataset of shirt images")
ap.add_argument("-t", "--query", required = True, help = "Path to the query image")
args = vars(ap.parse_args())

desc = LocalBinaryPatterns(24, 8)
index = {}

# loop over the shirt images
for imagePath in paths.list_images(args["dataset"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)

    # update the index dictionary
    filename = imagePath[imagePath.rfind("/")+1:]
    index[filename] = hist



# load the query image and extract LBPs from it
query = cv2.imread(args["query"])
queryFeatures = desc.describe(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))

# show the query image and initialize the results dictionary
cv2.imshow("Query", query)
results = {}

# loop over the index
for (k, features) in index.items():
    # compute the chi-squared distance
    d = 0.5*np.sum(((features-queryFeatures)**2/(features+queryFeatures+1e-10)))
    results[k] = d


# sort the results
results = sorted([(v, k) for (k, v) in results.items()])[:3]

#loop over the results
for (i, (score, filename)) in enumerate(results):
    print ("#%d. %s: %.4f"%(i+1,filename, score))
    image = cv2.imread(arg["dataset"]+"/", filename)
    cv2.imshow("Result #{}".format(i+1), image)
    cv2.waitKey(0)

"""
Pros:
- The original implementation of LBPs is not rotationally invariant, but their extensions to LBPs implemented in scikit-image are
- very good at characterizing the texture of an image.
- Useful in face recognition.

Cons:
- can easily lead to larger feature vectors if not useful.
- Computationally prohibitive as the number of points and radius increases.

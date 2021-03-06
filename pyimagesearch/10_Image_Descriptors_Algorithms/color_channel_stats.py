""" Color channel statistics

- Separate the input Image into its respective channels. For an RGB image, we want to examine each of the Red, Green, and blue channel respectively
- Compute various statistic for each channel, such as mean, standard deviation, skew and kurtosis.
- Concatenate the statistic together to form a list of statistics for each color channel - this becomes our feature vector
"""

from scipy.spatial import distance as dist
import imutils as paths
import numpy as np
import cv2, glob, os

foldername=""
imagePaths = glob.glob(os.getcwd()+"/"+foldername)
index = {}

#loop over the images
for imagepath in imagePaths:
    # load the image and extract the filename
    image = cv2.imread(imagepath)
    filename = imagapath.rsplit("/")[-1]

    # extract the mean and standard deviation from each channel of the BGR image, then update the index with the feature vector
    (means, stds) = cv2.meanStdDev(image)
    features = np.concatenate([means,stds]).flatten()
    index[filename]=features


#display the query image and grab the sorted keys of the index dictionary
query = cv2.imread(imagePaths[0])
cv2.imshow("Query (trex_01.png)", query)
keys = sorted(index.keys())

#loop over the filenames in the dictionary
for (i, k) in enumerate(keys):
    if k == "trex_01.png":
        continue
    #load the current image and compute the Euclidean distance between the query image and the current image
    image = cv2.imread(imagePaths[i])
    d = dist.euclidean(index["trex_01.png"], index[k])

    #display the distance between the query image and the current image
    cv2.putText(image,"%.2f" %(d), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow(k, image)

# wait for the keypress
cv2.waitKey(0)

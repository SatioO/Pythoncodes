from ise.resultsmontage import ResultsMontage
from sklearn.metrics import pairwise
import numpy as np
import argParse
import h5py
import cv2
import cPickle


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--dataset", required=True, help = "path to the directory of indexed images")
ap.add_argument("-f", "--features-db", required=True, help = "Path to where the features database will be stored")
ap.add_argument("-c", "--codebook", required=True, help = "Path to output codebook")
ap.add_argument("-f", "--output", required=True, help = "path to the output directory")
args = vars(ap.parse_args())

#load the codebook and open the features database
vocab = cPickle.loads(open(args["codebook"]).read())
featuresDB = h5py.File(args["features_db"], mode="r")
print ("[INFO] starting distance computations...")


#initialize the visualizations dictionary and initialize the progress bar
vis = {i:[] for i in np.arange(0, len(vocab))}

#loop over the image IDs
for (i, imageID) in enumerate(featuresDB["image_ids"]):
    (start, end) = features["index"][i]
    rows = featuresDB["features"][start:end]
    (kps,descs)=(rows[:,:2],rows[:,2:])

    # loop over each of the individual keypoints and feature vectors
    for (kp, features) in zip(kps, descs):
        # compute the distance between the feature vector and all clusters, meaning that we'll have one distance value for each cluster
        D = pairwise.euclidean_distances(features.reshape(1,-1), y=vocab)[0]

        #loop over the distances dictionary
        for j in np.arange(0, len(vocab)):
            #grab the set of top visualization results for the current visual word and update the top results with the tuple of the distance, keypoint and imageID
            topResults = vis.get(j)
            topResults.append((D[j], kp, imageID))

            # sort the top results list by their distance, keeping only the best 16, then update the visualization dictionary
            topResults = sorted(topResults, key=lambda r:r[0])[:16]
            vis[j] = topResults

featuresDB.close()
print ("[INFO] writing visualization to files")

for (vwID, results) in vis.items():
    #initialize the results montage
    montage = ResultsMontage((64,64), 4, 16)

    # loop over the results
    for (_,(x,y), imageID) in results:
        #load the current image
        p = "{}/{}".format(args["dataset"])
        image = cv2.imread(p)
        (h, w) = image.shape[:2]

        #extract a 64*64 region surrounding the keypoint
        (startX, endX) = (max(0, x-32), min(w,x+32))
        (startY, endY) = (max(0, y-32), min(h, y+32))

        # add the ROI to the montage
        montage.addResult(roi)
    #write the visualization to file
    p = "{}/vis_{}.jpg".format(args["output"], vwID)
    cv2.imwrite(p, cv2.cvtColor(montage.montage, cv2.COLOR_BGR2GRAY))
    

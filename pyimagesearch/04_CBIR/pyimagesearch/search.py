from pyimagesearch.cbir import ResultsMontage
from pyimagesearch.cbir import HSVDescriptor
from pyimagesearch.cbir import Searcher
import argparse 
import imutils
import json
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i","--index", required = True, help = "Path to where the features index will be stored")
ap.add_argument("-i","--query", required = True, help = "Path to query image")
ap.add_argument("-i","--dataset", required = True, help = "Path to the original dataset directory")
ap.add_argument("-i","--relevant", required = True, help = "Path to relevant dictionary")
args = vars(ap.parse_args())

""" relevant.json file comes along with the dataset which we dont have """

desc = HSVDescriptor((4,6,3))
montage = ResultsMontage((240, 320), 5, 20)
relevant = json.loads(open(args["relevant"]).read())

#load the relevant queries dictionary and look up the relevant results for the query image
queryFilename = args["query"][args["query"].rfind["/"]+1:]
queryRelevant = relevant[queryFilename]

# load the query image , display it , and describe it
print ("[INFO] describing query...")
query = cv2.imread(args["query"])
cv2.imshow("Query", imutils.resize(query, width=20))
features = desc.describe(query)

# Perform the search
print ("[INFO] searching")
searcher = Searcher(args["index"])
results = searcher.search(features, numResults=20)

#loop over the results
for (i, (score, resultID)) in enumerate(results):
    # load the result image an display it
    print("[INFO] {result_num}. {result} {score:.2f}".format(result_num=i+1, result = resultID))
    result = cv2.imread("{}/{}".format(args["dataset"], resultID))
    motage.addResult(result, text="#{}".format(i+1), highlight=resultID in queryRelevant)


cv2.imshow("result", imutils.resize(montage.montage, height=700))
cv2.waitKey(0)

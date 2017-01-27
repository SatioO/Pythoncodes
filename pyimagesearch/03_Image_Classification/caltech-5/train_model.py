from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
import numpy as np
import argparse, h5py, cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help = "Path to dataset")
ap.add_argument("-f", "--features-db", required=True, help = "Path to where the features data will be stored")
ap.add_argument("-b", "--bovw-db", required=True, help = "Path to where the bag of visual words are stored")
ap.add_argument("-b", "--model", required=True, help = "Path to the output classifier")
args = vars(ap.parse_args())

featuresDB = h5py.File(args["features_db"])
bovwDB = h5py.File(args["bovw_db"])

(trainData, trainLabels) = (bovwDB["bovw"][:300], featuresDB["image_ids"][:300])
(testData, testLabels) = (bovwDB["bovw"][300:], featuresDB["image_ids"][300:])

trainLabels = [l.split(":")[0] for l in trainLabels]
testLabels = [l.split(":")[0] for l in testLabels]

params = {"C":[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LinearSVC(random_state=42),params, cv=3)
model.fit(trainData, trainLabels)

predictions = model.predict(testData)
print(classification_report(testLabels, predictions))

for i in np.random.choice(np.arange(300,500), size=(20,), replace=False):
    (labels,filename)=featuresDB["image_ids"][i].split(":")
    image =cv2.imread("{}/{}/{}".format(args["dataset"], label, filename))
    prediction = model.predict(bovwDB["bovw"][i].reshape(1,-1))[0]

    #show the prediction
    print ("[PREDICTION] {}:{}".format(filename, prediction))
    cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0,255,0),2)
    cv2.imshow("Image",image)
    cv2.waitKey(0)

featuresDB.close()
bovwDB.close()

f = open(args["model"],"w")
f.write(cPickle.dumps(model))
f.close()

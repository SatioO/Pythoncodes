from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np

print ("[INFO] downloading mnist")

dataset = datasets.fetch_mldata("MNIST Original")

(trainData, testData, trainLabels, testLabels) = train_test_split( dataset.data/255.0, dataset.target.astype("int"), test_size=0.33)

dbn = DBN(
[trainData.shape[1],300,10],
learn_rates=0.3,
learn_rate_decays=0.9,
epochs=10,
verbose=1)
dbn.fit(trainData, trainLabels)

predictions = dbn.predict(testData)
print(classification_report(testLabels, predictions))

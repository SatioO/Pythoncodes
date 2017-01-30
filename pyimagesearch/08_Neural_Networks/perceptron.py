from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn import datasets

iris = datasets.load_iris()
(trainData, testData, trainLabels, testLabels) = train_test_split(iris.data, iris.labels, test_size = 0.25, random_state= 42)

#train the perceptron
model = Perceptron(n_iter=10, eta0=1.0, random_state=42)
model.fit(trainData,trainLabels)

#evalute the performance
predictions = model.predict(testData)
print (classification_report(predictions, testLabels, target_names = iris.target_names))

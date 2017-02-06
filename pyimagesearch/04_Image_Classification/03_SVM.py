"""
- Basic Concpets of SVM.
- Concpets of "linear separability" and "maximum margin".
- XOR Problem by utilizing kernels to achieve non-linear separation.
"""

# Gram Matrix [ Kernel Matrix]
#A' = [K(A,A), K(A,B), K(A,C), K(A,D)]
#B' = [K(B,A), K(B,B), K(B,C), K(B,D)]
#C' = [K(C,A), K(C,B), K(C,C), K(C,D)]
#D' = [K(D,A), K(D,B), K(D,C), K(D,D)]

"""
#Types of Kernels
#Linear
K(x,y) = x(T)*y

#Polynomial
K(x,y) = (x(T)*y+c)^d

#Sigmoid
K(x,y) = tanh(gamma*x(T)*y+c)

#Radial Basis function(RBF):
K(x,y) = exp(-gamma*||x-y||^2)
"""

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np

# generate the XOR data
tl = np.random.uniform(size=(100, 2)) + np.array([-2.0, 2.0])
tr = np.random.uniform(size=(100, 2)) + np.array([2.0, 2.0])
br = np.random.uniform(size=(100, 2)) + np.array([2.0, -2.0])
bl = np.random.uniform(size=(100, 2)) + np.array([-2.0, -2.0])
X = np.vstack([tl, tr, br, bl])
y = np.hstack([[1] * len(tl), [-1] * len(tr), [1] * len(br), [-1] * len(bl)])

# construct the training and testing split by taking 75% of the data for training
# and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(X, y, test_size=0.25,
	random_state=42)

# train the linear SVM model, evaluate it, and show the results
print("[RESULTS] SVM w/ Linear Kernel")
model = SVC(kernel="linear")
model.fit(trainData, trainLabels)
print(classification_report(testLabels, model.predict(testData)))
print("")

# train the SVM + poly. kernel model, evaluate it, and show the results
print("[RESULTS] SVM w/ Polynomial Kernel")
model = SVC(kernel="poly", degree=2, coef0=1)
model.fit(trainData, trainLabels)
print(classification_report(testLabels, model.predict(testData)))

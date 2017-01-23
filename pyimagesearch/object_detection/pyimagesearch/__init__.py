"""


- 6 step framework
   - Sample P Positive samples from your training data of the object(s) you want to detect, and extract HOG descriptors from these examples
   -  Sample N negitive samples from a negitive training set that does not contain any of the objects you want to detect and extract HOG descriptors from these samples as well N >>> P
   - Train a linear SVM on your positive and negative examples
   - Apply hard negative mining - track all the mis-classified feature vectors and use them as negative examples
   - Take the false positive samples found during the hard negative mining stage, sort them by confidence
   and retrain your classifier using these hard-negative samples
   - Classifier is trained and apply your test dataset. Apply non-maxima suppression to remove reduntant and overlapping bounding boxes.


   #1  Experiment Preparation
   #2  Feature extraction
   #3  Detector training
   #4  Non-maxima suppression
   #5  Hard negative mining
   #6  Detection retraining

"""


""" # Framework

| --- pyimagesearch
|     | --- __int__.py
|     | --- descriptors
           | --- __init__.py
           | --- hog.py
      | --- object_detection
           | --- __init__.py
           | --- helper.py
           | --- nms.py
           | --- objectdetector.py
      | --- utils
           | --- __init__.py
           | --- conf.py
           | --- dataset.py
| --- explore_dims.py
| --- extract_features.py
| --- hard_negative_mine.py
| --- test_model.py
| --- train_model.py
"""

""" # script for accessing h5py files
import h5py
db = h5py.File("car_features.hdf5")#load the file

[i for in db.keys()] # for available datasets

db["features"].shape # shape of the dataframe

row = db["features"][0] # extracting 1st row

(label, features) = (row[0], row[1:])

label

features.shape

"""


""" Tips on training your own image detector

- Take special care labeling your data
   "Garbage in, Garbage out"
- Leverage Parallel Processing
- Use dlib as a starting point
- keep in mind the image pyramid and sliding window tradeoff
- Tune detector hyperparameters
- Run experiments and log your results

"""

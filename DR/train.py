"""

Training the model 

X_train - 34000
X_valid - 1000 (do stratified sampling)



For every epoch
     - Suffle all Image location 
     - Scale the Images to 270*270
     - Read a bunch of 8 Images using statrified sampling 
     - Randomly select the undersampled Images and rotate them and stack them - 4
     - Randomly flip them and stack again - 4 


34000/8 = 4300 rounds create one epoch

run for 10 epoch



### Testing Images 
- Send each Image 10 times by
  - Rotating +45,-45,+90,-90
  - Translation - -50,-50
  - Flipping - Horizantally, Vertically


"""

from helper import *
from model import *
import pandas as pd 
import numpy as np 
import cv2
import glob
import sklearn.cross_validation import train_test_split 


## Loads the labels 
labels = pd.read_table("/data/dr/trainLabels.csv",sep=",",index_col=["image"])
labels.drop("420_right")



filelist = glob.glob("/data/dr/data/sample_270_270/*.jpeg")

#Stratified sampling for dividing data into X_train and X_valid 
x_train,x_valid,y_train, y_valid = train_test_split(filelist,labels,
  test_size = 0.10, random_state = 20)


## train_test split 
x_train,x_test,y_train, y_test = train_test_split(x_train,y_train,
  test_size = 0.10, random_state = 20)


#read the x_valid and y_valid images 
x_valid = np.array([cv2.imread("/data/dr/data/sample_270_270/"+i+".jpeg") for i in y_valid.index])
y_valid = np.array(pd.get_dummies(y_valid["level"]))

"""

Once you get x_train,x_test and x_valid. Train the data as you need and test on validation 

"""

epoch = 10

train_acc = []
val_acc = []


for ep in range(epoch):
	concat_df = random_split(labels)
	for batch in range(len(concat_df)):
		X,Y = Image_generator(concat_df[batch])
		np.random.shuffle(list(zip(X,Y)))
		loss = model_train.train_on_batch(X,Y)
		test_loss = model_train.test_on_batch(x_valid,x_valid)
		print(loss)

		
# We need to shift from train_on_batch to fit_generator as this is more flexible and reduces your code to one line from present 
# 10+ lines . It also helps in reading Images on cpu and training network on GPU parallely, which saves a lot of time.

# Load the required files 
import numpy as np 
import pandas as pd 
import cv2


"""
List of functions available on this page 

rotate - rotates the Image by the specified angle.
translate- will translate the image by the specified (x,y) lengths 
random_split 
      - takes input as pandas Dataframe and a column to be split
      - outputs a list of n-dataframes (each dataframe have the class outcome in the same ratio)
data_generator
      - takes input as pandas Dataframe and location of the Images 
      - outputs a tuple(X,Y) (X-Images, Y-labels)
Image_reader
      - Does the same as data_generator but doesn't yield the output, instead return a set of Images
      - Useful for validation set 
testImage_reader
      - Takes an Image as input and outputs a numpy array of n-images (n-1 images are transformations here)
      
"""

## define a few functions 
def rotate(image, angle, center=None, scale=1.0):
	"""
	Rotates the Image by a specified angle 
	
	Args:
	image  - an numpy array of the image 
	angle  - Angle of rotation
	center - Default None and when used will use Image center for rotation
	scale  - to scale down the image if required (0-1)
	
	Returns:
	returns an numpy array of rotated Images 
	
	"""
	(h,w) = image.shape[:2]
	if center is None:
		center =(w/2,h/2)
		M = cv2.getRotationMatrix2D(center,angle,scale)
		rotated = cv2.warpAffine(image,M,(w,h))
	return rotated


def translate(image, x, y):
	"""
	translate the Image on the horizontal by x and vertical by y
	
	Args:
	image - an numpy array of an Image
	x     - translate the image horizantally by x units 
	y     - translate the image vertically by y units 
	
	Returns:
	returns an numpy array of shifted image
	
	"""
	M = np.float32([[1,0,x],[0,1,y]])
	shifted = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
	return shifted


# create a list of individual groups 
def random_split(dataframe, colname = "level"):
	"""
	create a list of n-dataframes with same proportions of class labels across all the dataframes
	
	Args:
	dataframe - a pandas dataframe
	colname   - column name of which you want to segment into n-list 
	
	Returns:
	A list with n-dataframes 
	
	"""
	gb = dataframe.groupby(colname)
	groups = [gb.get_group(x) for x in gb.groups]
	# Get the ratio of Images with most under-represented class 
	value_counts = np.array(dataframe[colname].value_counts())
	ratio = value_counts/min(value_counts)
	tot_groups = value_counts/ratio
	# Divide each group into those many splits
	group_split = [np.array_split(groups[x],tot_groups[x]) for x in range(len(groups))] 
	[np.random.shuffle(group_split[i]) for i in range(len(group_split))]
	min_group = min([len(group_split[i]) for i in range(len(group_split))])
	concat_df = [pd.concat([group_split[i][j] for i in range(len(group_split))]) for j in range(min_group)]
	return concat_df


def data_generator(labels, location = "/data/dr/data/sample_270_270/"):
	"""
	For oversampling the data and act as a generator to feed to fit_generator function for training the nn.
	
	Args:
	labels   - pandas dataframe of Image labels 
	location - location of where your Images are located. the default is set randomly
	
	Returns:
	yields a tuple (X,Y) everytime it is called.
	 X - numpy array of n-images
	 Y - respective labels of n-images which are modified to dummies 
	
	"""
	concat_df = random_split(labels)
	while True:
		for batch in range(len(concat_df)):
			i_all = np.array([cv2.imread(location+i+".jpeg") for i in pdframe.index])
			
			i_4 = np.array([cv2.imread(location+i+".jpeg") for i in pdframe.query("level == 4").index])
			i_4_rotate = np.array([rotate(i_4[j],i) for j in range(len(i_4)) for i in [30,60,45,150,135,210,240,225,300]])
			i_4_flip =np.array([cv2.flip(i_4_rotate[j],i) for j in range(len(i_4_rotate)) for i in [0,1]])
			
			i_3 = np.array([cv2.imread(location+i+".jpeg") for i in pdframe.query("level == 3").index])
			i_3_rotate = np.array([rotate(i_3[j],i) for j in range(len(i_3)) for i in [30,60,45,150,135,210,240,225,300]])
			i_3_flip =np.array([cv2.flip(i_3_rotate[j],i) for j in range(len(i_3_rotate)) for i in [0,1]])
			
			i_2 = np.array([cv2.imread(location+i+".jpeg") for i in pdframe.query("level == 2").index])
			i_2_rotate = np.array([rotate(i_2[j],i) for j in range(len(i_2)) for i in [60,150]])
			i_2_flip =np.array([cv2.flip(i_2_rotate[j],i) for j in range(len(i_2_rotate)) for i in [0,1]])
			
			i_1 = np.array([cv2.imread(location+i+".jpeg") for i in pdframe.query("level == 1").index])
			i_1_rotate = np.array([rotate(i_1[j],i) for j in range(len(i_1)) for i in [30,60,150,210,240,300]])
			i_1_flip =np.array([cv2.flip(i_1_rotate[j],i) for j in range(len(i_1_rotate)) for i in [0,1]])
			
			# now stack the Images using np.stack
			X = np.concatenate((i_all,i_4,i_4_rotate,i_4_flip,i_3,i_3_rotate,i_3_flip,i_2,i_2_rotate,i_2_flip,i_1,i_1_rotate,i_1_flip))
			Y = np.concatenate(((pdframe.values.reshape(len(pdframe))),
					   np.full(len(i_4)+len(i_4_rotate)+len(i_4_flip),4),
					   np.full(len(i_3)+len(i_3_rotate)+len(i_3_flip),3),
					   np.full(len(i_2)+len(i_2_rotate)+len(i_2_flip),2),
					   np.full(len(i_1)+len(i_1_rotate)+len(i_1_flip),1)))
			Y = np.array(pd.get_dummies(Y))
			np.random.shuffle(list(zip(X,Y)))
			yeild((X,Y))


def Image_reader(pdframe, location = "/data/dr/data/sample_270_270/"):
	
	"""
	For oversampling the data and act as a feeder to the train_on_batch function in keras 
	
	Args:
	pdframe   - pandas dataframe of Image labels 
	location - location of where your Images are located. the default is set randomly
	
	Returns:
	yields a tuple (X,Y) everytime it is called.
	 X - numpy array of n-images
	 Y - respective labels of n-images which are modified to dummies 
	
	"""
	i_all = np.array([cv2.imread(location+i+".jpeg") for i in pdframe.index])

	i_4 = np.array([cv2.imread(location+i+".jpeg") for i in pdframe.query("level == 4").index])
	i_4_rotate = np.array([rotate(i_4[j],i) for j in range(len(i_4)) for i in [30,60,45,150,135,210,240,225,300]])
	i_4_flip =np.array([cv2.flip(i_4_rotate[j],i) for j in range(len(i_4_rotate)) for i in [0,1]])


	i_3 = np.array([cv2.imread(location+i+".jpeg") for i in pdframe.query("level == 3").index])
	i_3_rotate = np.array([rotate(i_3[j],i) for j in range(len(i_3)) for i in [30,60,45,150,135,210,240,225,300]])
	i_3_flip =np.array([cv2.flip(i_3_rotate[j],i) for j in range(len(i_3_rotate)) for i in [0,1]])

	i_2 = np.array([cv2.imread(location+i+".jpeg") for i in pdframe.query("level == 2").index])
	i_2_rotate = np.array([rotate(i_2[j],i) for j in range(len(i_2)) for i in [60,150]])
	i_2_flip =np.array([cv2.flip(i_2_rotate[j],i) for j in range(len(i_2_rotate)) for i in [0,1]])

	i_1 = np.array([cv2.imread(location+i+".jpeg") for i in pdframe.query("level == 1").index])
	i_1_rotate = np.array([rotate(i_1[j],i) for j in range(len(i_1)) for i in [30,60,150,210,240,300]])
	i_1_flip =np.array([cv2.flip(i_1_rotate[j],i) for j in range(len(i_1_rotate)) for i in [0,1]])

	# now stack the Images using np.stack
	X = np.concatenate((i_all,i_4,i_4_rotate,i_4_flip,i_3,i_3_rotate,i_3_flip,i_2,i_2_rotate,i_2_flip,i_1,i_1_rotate,i_1_flip))
	Y = np.concatenate(((pdframe.values.reshape(len(pdframe))),
		np.full(len(i_4)+len(i_4_rotate)+len(i_4_flip),4),
		np.full(len(i_3)+len(i_3_rotate)+len(i_3_flip),3),
		np.full(len(i_2)+len(i_2_rotate)+len(i_2_flip),2),
		np.full(len(i_1)+len(i_1_rotate)+len(i_1_flip),1)))
	Y = np.array(pd.get_dummies(Y))
	return X,Y



# both the generators are working . I prefer using the data_generator.

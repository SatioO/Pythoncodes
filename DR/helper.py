"""

Image Batch Generator for Imbalanced datasets 



 - Split the labels into groups 
 - Make the count of each group 
      - 0 - 
      - 1 - 
      - 2 - 
      - 3 - 
      - 4 - 
 -  With the most under-represented class count, divide all other classes count and keep a count of each. Take ratio virtually.
 -  Now randomly suffle each class and create a split of len(class)/ratio(class/most under-represented class).
 - Split the dataframes accordingly and stack them accordingly
"""

"""

- Divide the dataframe into n-groups(n-classes). Each group containing only one class


"""

## define a few functions 
def rotate(image,angle,center=None,scale=1.0):
	(h,w) = image.shape[:2]
	if center is None:
		center =(w/2,h/2)
		M = cv2.getRotationMatrix2D(center,angle,scale)
		rotated = cv2.warpAffine(image,M,(w,h))
	return rotated


def translate(image,x,y):
	M = np.float32([[1,0,x],[0,1,y]])
	shifted = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
	return shifted


# create a list of individual groups 
def random_split(dataframe,colname = "level"):
	gb = dataframe.groupby(colname)
	groups = [gb.get_group(x) for x in gb.groups]
	# Get the ratio of Images with most under-represented class 
	value_counts = np.array(labels[colname].value_counts())
	ratio = value_counts/min(value_counts)
	tot_groups = value_counts/ratio
	# Divide each group into those many splits
	group_split = [np.array_split(groups[x],tot_groups[x]) for x in range(len(groups))] 
	[np.random.shuffle(group_split[i]) for i in range(len(group_split))]
	min_group = min([len(group_split[i]) for i in range(len(group_split))])
	concat_df = [pd.concat([group_split[i][j] for i in range(len(group_split))]) for j in range(min_group)]
	return concat_df





def Image_generator(pdframe):
	i_all = np.array([cv2.imread("/data/dr/data/sample_270_270/"+i+".jpeg") for i in pdframe.index])

	i_4 = np.array([cv2.imread("/data/dr/data/sample_270_270/"+i+".jpeg") for i in pdframe.query("level == 4").index])
	i_4_rotate = np.array([rotate(i_4[j],i) for j in range(len(i_4)) for i in [30,60,45,150,135,210,240,225,300]])
	i_4_flip =np.array([cv2.flip(i_4_rotate[j],i) for j in range(len(i_4_rotate)) for i in [0,1]])


	i_3 = np.array([cv2.imread("/data/dr/data/sample_270_270/"+i+".jpeg") for i in pdframe.query("level == 3").index])
	i_3_rotate = np.array([rotate(i_3[j],i) for j in range(len(i_3)) for i in [30,60,45,150,135,210,240,225,300]])
	i_3_flip =np.array([cv2.flip(i_3_rotate[j],i) for j in range(len(i_3_rotate)) for i in [0,1]])

	i_2 = np.array([cv2.imread("/data/dr/data/sample_270_270/"+i+".jpeg") for i in pdframe.query("level == 2").index])
	i_2_rotate = np.array([rotate(i_2_rotate[j],i) for j in range(len(i_2)) for i in [60,150]])
	i_2_flip =np.array([cv2.flip(i_2_rotate[j],i) for j in range(len(i_2_rotate)) for i in [0,1]])

	i_1 = np.array([cv2.imread("/data/dr/data/sample_270_270/"+i+".jpeg") for i in pdframe.query("level == 1").index])
	i_1_rotate = np.array([rotate(i_1[j],i) for j in range(len(i_1)) for i in [30,60,150,210,240,300]])
	i_1_flip =np.array([cv2.flip(i_1_rotate[j],i) for j in range(len(i_1_rotate)) for i in [0,1]])

	# now stack the Images using np.stack
	X = np.concatenate((i_all,i_4,i_4_rotate,i_4_flip,i_3,i_3_rotate,i_3_flip,i_2,i_2_rotate,i_2_flip,i_1,i_1_rotate,i_1_flip))
	Y = np.concatenate((pdframe.values.reshape(len(pdframe)),
		np.full(len(i_4,i_4_rotate,i_4_flip),4),
		np.full(len(i_3,i_3_rotate,i_3_flip),3),
		np.full(len(i_2,i_2_rotate,i_2_flip),2),
		np.full(len(i_1,i_1_rotate,i_1_flip),1)))






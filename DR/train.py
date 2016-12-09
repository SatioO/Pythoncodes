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


import pandas as pd 
import numpy as np


from sklearn.cross_validation import StratifiedKFold


filelist = glob.glob("/Users/Satish/Downloads/DR/train/*.jpeg") 


# Resizing Image to (270,270)
h = 270 
for i in filelist:
	img = cv2.imread(i)
	l = img.shape[0]*h/img.shape[1]
	eye = cv2.resize(img,(h,l))
	background = numpy.full((270,270,3),128)
	background[135-eye.shape[0]/2:135+eye.shape[0]/2, 135-eye.shape[1]/2:135+eye.shape[1]/2,:] = eye
	filename = "/Users/Satish/Downloads/DR/kaggle_sol/"+i.rsplit('/',1)[-1]
	cv2.imwrite(filename,background)


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


flipped = cv2.flip(image,0) # Horizantally
flipped = cv2.flip(image,1) # verfically



## Loads the labels 
labels = pd.read_table("/data/dr/trainLabels.csv",sep=",",index_col=["image"])
labels.drop(labels.index[["194_left.jpeg"]])

# Make a list of Image Index
filelist = ["/data/dr/data/pre_process/"+i+".jpeg" for i in labels.index]

#Stratified sampling for dividing data into X_train and X_valid 
eval_size = 0.1
KF = StratifiedKFold(y,round(1./eval_size))
train_indicies , valid_indicies = next(iter(KF))

x_train , y_train = filelist.ix[train_indicies],labels[train_indicies]
x_valid , y_valid = filelist.ix[valid_indicies],labels[valid_indicies]


## train_test split 
x_train,x_test,y_train, y_test = train_test_split(x_train,y_train,
  test_size = 0.10, random_state = 20)




"""

Once you get x_train,x_test and x_valid. Train the data as you need and test on validation 

"""

epoch = 10 
batch_size = 16

train_acc = []
val_acc = []

for ep in range(epochs):
	np.random.shuffle(len(zip(x_train,y_train)))
	x_image, y_label = Image_Generator(x_train)



from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=(len(labels)/16))



# once after getting the list 
def Image_Generator(y_train,batch_size):
	np.random.shuffle(y_train)
	length = len(y_train)//batch_size
	for i in range(length):
		y_label =y_train.groupby('level', as_index=False).apply(lambda obj: obj.loc[np.random.choice(obj.index, size,  True)]).reset_index()
		batch = filelist[i*batch_size:(i+1)*batch_size]
		Image = [cv2.imread(i) for i in batch] # read the Image 





def batch_generator(X,y,batch_size,shuffle):
	number_of_batches = np.ceil(len(X)/batch_size)
	couter = 0
	sample_index = np.arange(len(X))
	if shuffle:
		np.random.shuffle(sample_index)
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]



def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0
		


def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0





ImageFile.LOAD_TRUNCATED_IMAGES = True # 

 # resize all the images and save them in another folder 
x = 0
size = (270, 270)
for i in filelist:
	im = Image.open(i)
	im.thumbnail(size, Image.ANTIALIAS)
	background = Image.new('RGB', size)
	background.paste(im,((size[0] - im.size[0]) / 2, (size[1] - im.size[1]) / 2))
	filename = "/Users/Satish/Downloads/DR/pre_process/"+i.rsplit('/',1)[-1]
	background.save(filename,"JPEG")
	x=x+1
	print (x)


# Labels of the Images 
lables = pd.read_table("/Users/Satish/Downloads/DR/trainLabels.csv",sep=",",index_col = ["image"])

filelist = glob.glob("/Users/Satish/Downloads/DR/pre_process/*.jpeg") 
## Separate the left eye Images with the right eye Images 
left_eye = [x for x in filelist if re.search(r'left',x)]
right_eye = [x for x in filelist if re.search(r'right',x)]


## Here the Images are of small Size  - valid if we are doing regression(cosidering the outcome to be ordinal variable)
left_eye_label  = [float(lables.ix[x.rsplit("/",1)[-1].rsplit(".")[0]]) for x in left_eye]
right_eye_label = [float(lables.ix[x.rsplit("/",1)[-1].rsplit(".")[0]]) for x in right_eye]
lr_label = [max(a,b) for a,b in zip(left_eye_label,right_eye_label)]










X_train,X_val,train_Y, val_Y = train_test_split(zip(left_eye,right_eye),lr_label,
	test_size = 0.10, random_state = 20)

le_val = np.array([io.imread(X_val[i][0]) for i in np.arange(len(X_val))])
re_val = np.array([io.imread(X_val[i][1]) for i in np.arange(len(X_val))])
val_Y = np.array(val_Y)

epochs = 10 
batch_size = 128 

## Training the neural network 

model_train = model(input_shape)


for ep in range(epochs):
	np.random.shuffle(zip(X_train,train_Y))
	n_batches = len(X_train)//batch_size
	for batch_i in range(n_batches):
		X = X_train[batch_i * batch_size: (batch_i + 1) * batch_size] # select the batch of Images
		leX = np.array([io.imread(X[i][0]) for i in np.arange(len(X))])
		reX = np.array([io.imread(X[i][1]) for i in np.arange(len(X))])
		out = np.array(train_Y[batch_i * batch_size: (batch_i + 1) * batch_size])
		loss = model_train.train_on_batch([leX,reX],out)
		test_loss = model_train.test_on_batch([le_val,re_val],val_Y)
		print (ep,batch_i,loss,test_loss)
	score = model_train.evaluate([le_val,re_val],val_Y,verbose=0)
	print ("Test Score",score)

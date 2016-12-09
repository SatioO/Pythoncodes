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




## Loads the labels 
labels = pd.read_table("/data/dr/trainLabels.csv",sep=",",index_col=["image"])
labels.drop("420_right")



filelist = glob.glob("/data/dr/data/sample_270_270/*.jpeg")

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

train_acc = []
val_acc = []


for ep in range(epoch):
	concat_df = random_split(labels)
	for batch in range(len(concat_df)):
		X,Y = Image_generator(concat_df[batch])
		loss = model_train.train_on_batch(X,Y)
		test_loss = model_train.test_on_batch(X_valid,Y_valid)


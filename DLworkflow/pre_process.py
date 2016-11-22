filelist = glob.glob("/Users/Satish/Downloads/DR/train/*.jpeg") 

ImageFile.LOAD_TRUNCATED_IMAGES = True # 

 # resize all the images and save them in another folder 
x = 0
size = (64, 64)
for i in filelist:
	im = Image.open(i)
	im.thumbnail(size, Image.ANTIALIAS)
	background = Image.new('RGBA', size)
	background.paste(im,((size[0] - im.size[0]) / 2, (size[1] - im.size[1]) / 2))
	filename = "/Users/Satish/Downloads/DR/pre_process/"+i.rsplit('/',1)[-1]
	background.save(filename,"JPEG")
	x=x+1
	print (x)


# Labels of the Images 
lables = pd.read_table("/Users/Satish/Downloads/DR/trainLabels.csv",sep=",",index_col = ["image"])




## Separate the left eye Images with the right eye Images 
left_eye = [x for x in filelist if x.rsplit("/",1)[-1].rsplit(".")[0].rsplit("_")[-1] == "left" ]
right_eye = [x for x in filelist if x.rsplit("/",1)[-1].rsplit(".")[0].rsplit("_")[-1] == "right" ]


## Here the Images are of small Size  - valid if we are doing regression(cosidering the outcome to be ordinal variable)
left_eye_label  = [float(lables.ix[x.rsplit("/",1)[-1].rsplit(".")[0]]) for x in left_eye]
right_eye_label = [float(lables.ix[x.rsplit("/",1)[-1].rsplit(".")[0]]) for x in right_eye]
lr_label = [max(a,b) for a,b in zip(left_eye_label,right_eye_label)]



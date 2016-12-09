## Kaggle winning solution 


"""

IMAGE PRE-PROCESSING 

"""

import cv2,glob,numpy

def scaleRadius(img,scale):
	x = img[img.shape[0]/2,:,:].sum(1)
	r = (x>x.mean()/10).sum()/2
	s = scale*1.0/r
	return cv2.resize(img,(0,0),fx=s,fy=s)


scale = 270
for f in glob.glob("/Users/Satish/Downloads/DR/train/*.jpeg"):
	try:
		a = cv2.imread(f)
		# scale the image to given radius 
		a = scaleRadius(a,scale)
		# subtract local mean color
		a = cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)
		#remove the outer 10%
		b = numpy.zeros(a.shape)
		cv2.circle(b,(a.shape[1]/2,a.shape[0]/2),int(scale*0.9),(1,1,1),-1,8,0)
		a = a*b+128*(1-b)
		filename = "/Users/Satish/Downloads/DR/kaggle_sol/"+f.rsplit('/',1)[-1]
		cv2.imwrite(filename,a)
		print 0 
	except:
		print f 




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

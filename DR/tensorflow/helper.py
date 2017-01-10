"""
Random Affine Transformation - WarpAffine function
Randomly cropped to 85-95%
Horizantally flipped
rotated between 0 and 360 degrees and then
scaled to the desired model input size

Channel-wise global contrast normalization was applied to normalize image color

PreLU weight initialization
Leaky rectifier non-linearities
Nestrov Momentum of 0.9

training:
3-5 10^-4 to 10^-5
0.003 - 100 epochs
0.001 for 30 epochs
0.0001 for 20 epochs
"""
import glob, cv2 , random
import numpy as np
import pandas as pd


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

def img_location_reader(folder_loc = "/Users/Satish/Downloads/DR/train"):
    """
    Returns a list of lists with each list containing the locations of each class

    Args:
    folder_loc : A string which contains Image folder

    """
    return glob.glob(folder_loc+"/*.jpeg")

def img_location_list(labels, folder_loc="/Users/Satish/Downloads/DR/train", over_sample = True):
    """
    Returns a list of image locations

    Args:
    folder_loc = location of the images. A string
    labels = a pandas dataframe with index as image name and one column name as level
    over_sample = if True , will oversample and balance the class.
    """
    if over_sample == True:
        max_class = max(labels.level.value_counts())
        fileloc = [list(folder_loc+labels[(labels.level == i)].index+".jpeg") for i in range(5)]
        img_list =  [random.choice(fileloc[i]) for i in range(len(fileloc)) for j in range(max_class)]

    else:
        img_list = [folder_loc+labels.index[i]+".jpeg" for i in range(len(labels.index))]

    return img_list

def image_read(image_location,labels,col_name="level"):
    """
    A function to convert the mode of GBR to RGB mode
    Args:
    image_location = input the Image location.
    Returns:
    return a numpy array in RGB mode
    """
    x = cv2.imread(image_location)
    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    y = image_location.rsplit("/")[-1].rsplit(".")[0]
    y = labels.loc[y,col_name]
    return x,y

def agument_data(image):
    """
    returns a numpy array of shape NHWC.

    Args:
    image: A numpy array of an image shape(HWC).
    """
    rotation = rotate(image,random.choice(range(360)))
    flip_h = cv2.flip(image,1)
    flip_v = cv2.flip(image,0)
    agumented_data = np.concatenate((image[np.newaxis,:,:,:],rotation[np.newaxis,:,:,:],flip_h[np.newaxis,:,:,:],flip_v[np.newaxis,:,:,:]))
    return agumented_data

def contrast_channel_wise(img):
    """
    Channel-wise global contrast normalizatin was applied to normalize image color
    Args:
    img: A numpy 3D array image

    Returns:
    A numpy array with channel wise global contrast normalization applied
    """
    x = (img[:,:,0]-(img[:,:,0].reshape(-1).mean()))/(img[:,:,0].reshape(-1).std())
    y = (img[:,:,1]-(img[:,:,1].reshape(-1).mean()))/(img[:,:,1].reshape(-1).std())
    z = (img[:,:,2]-(img[:,:,2].reshape(-1).mean()))/(img[:,:,2].reshape(-1).std())
    xyz = np.concatenate((x[np.newaxis,:,:],y[np.newaxis,:,:],z[np.newaxis,:,:]))
    xyz = np.moveaxis(xyz,0,-1)
    return xyz

def resize(image,size = (724,724)):
    """
    resize the image to required size

    Args:
    image: A numpy 3D array of image
    size: required size of an image. A tuple

    Returns:
    Numpy array of a resized Image

    """
    h = size[0]
    l = image.shape[1]*h/image.shape[0]
    resize = cv2.resize(image,(l,h),interpolation=cv2.INTER_AREA)
    num = random.choice(range(resize.shape[1]-h))
    resize = resize[:,num:num+h,:]
    return resize

def dev_image_reader(filelist,image_dummy_label):
    images = []
    images_label = []
    for j in range(len(filelist)):
        x_image,y_label= image_read(filelist[j])
        x_images = agument_data(x_image)
        x_images = np.concatenate([contrast_channel_wise(x_images[i])[np.newaxis,:,:,:] for i in range(len(x_images))])
        x_images = np.concatenate([resize(x_images[i],size=(724,724))[np.newaxis,:,:,:] for i in range(len(x_images))])
        images.append(x_images)
        images_label.append([img_dummy_label[y] for x in range(len(x_images))])
        x_image,y_image = np.concatenate(images),np.concatenate(images_label)
    return x_image,y_image


def val_test_image_reader(filelist):
    img_list = img_location_list(filelist,over_sample=False)
    img_dummy_label = pd.get_dummies(labels["level"].unique())
    images = []
    images_label = []
    for j in range(len(img_list)):
        x_image,y_label= image_read(img_list[j])
        x_image = contrast_channel_wise(x_image)
        x_image = resize(x_image, size=(724,724))
        images.append(x_image[np.newaxis,:,:,:])
        images_label.append([img_dummy_label[y_label] for x in range(len(x_image[np.newaxis,:,:,:]))])
    x_image,y_image = np.concatenate(images),np.concatenate(images_label)
    return x_image,y_image

# def train_test_split(imglist,test_size=0.3, random_state = 0):
#     """
#     returns train and test list with Image_locations
#
#     Args:
#     imglist = list of Image_locations
#     test_size = Proportion of Images to be hold for test . Default = 0.3
#     random_state = to replicate the results. Default = 0
#     """
#     random.seed(random_state)
#     sample = random.sample(xrange(len(imglist)), int((1-test_size)*(len(imglist))))
#     train = [imglist[i] for i in sample]
#     test = [x for x in imglist if x not in train]
#     return train,test

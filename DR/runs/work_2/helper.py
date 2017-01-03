import os
import glob
from tqdm import tqdm
import random
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import cv2



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



def img_location_reader(foldername):
    """
    Returns a list of lists with each list containing the locations of each class

    Args:
    foldername : string of the foldername where the images reside

    """
    folderlist = glob.glob(os.getcwd()+"/"+foldername+"/*")
    imglist = [glob.glob(i+"/*") for i in folderlist]
    return imglist


def train_test_split(imglist,test_size=0.3, random_state = 0):
    random.seed(random_state)
    test = []
    train = []
    for i in range(len(imglist)):
        x = [random.choice(imglist[i]) for j in range(int(len(imglist[i])*test_size))]
        y = [m for m in imglist[i] if m not in x]
        test.append(x)
        train.append(y)
    return train,test


def img_location_list(imglist,over_sample = True):
    """
    Returns a list of image locations

    Args:
    imagelist = a list of lists with each list containing the location of each class
    over_sample = if True , will oversample and balance the class.
    """
    if over_sample == True:
        max_class = int(max([len(imglist[i])] for i in range(len(imglist)))[0])
        img_list =  [random.choice(imglist[i]) for i in range(len(imglist)) for j in range(max_class)]
    else:
        img_list = [item for sublist in imglist for item in sublist]

    return img_list


def image_read(image_location,resize=True, size = (256,256)):
    """
    A function to convert the mode of GBR to RGB mode
    Args:
    image_location = input the Image location.
    resize = if True will resize the image
    Returns:
    return a numpy array in RGB mode
    """
    x = cv2.imread(image_location)
    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    y = image_location.rsplit("/")[-2]
    if resize == True:
        x = cv2.resize(x,size, interpolation = cv2.INTER_CUBIC)
    return x,y

def agument_data(image):
    """
    returns a numpy array of shape NHWC.

    Args:
    image: A numpy array of an image shape(HWC).
    """
    rotation = rotate(image,90*random.choice([1,2,3,4]))
    flip_h = cv2.flip(image,1)
    flip_v = cv2.flip(image,0)
    agumented_data = np.concatenate((image[np.newaxis,:,:,:],rotation[np.newaxis,:,:,:],flip_h[np.newaxis,:,:,:],flip_v[np.newaxis,:,:,:]))
    return agumented_data


def image_label(imglist):
    """
    Returns a dictionary with keys as image label and value as number of images

    Args:
    imglist: A list with image locations

    """
    img_dict={}
    for i in range(len(imglist)):
        y = imglist[i].rsplit("/")[-2]
        if y not in img_dict.keys():
            img_dict[y] = 1
        else:
            img_dict[y] += 1
    return img_dict


def dev_image_reader(filelist,img_dummy_label):
    images = []
    images_label = []
    for j in range(len(filelist)):
        x_image,y_label= image_read(filelist[j])
        x_images = agument_data(x_image)
        images.append(x_images)
        images_label.append([img_dummy_label[y_label] for x in range(len(x_images))])
        x_image,y_image = np.concatenate(images),np.concatenate(images_label)
    return x_image,y_image


def val_test_image_reader(filelist):
    img_list = img_location_list(filelist,over_sample=False)
    img_dummy_label = pd.get_dummies(list(image_label(img_list).keys()))
    images = []
    images_label = []
    for j in range(len(img_list)):
        x_image,y_label= image_read(img_list[j])
        images.append(x_image[np.newaxis,:,:,:])
        images_label.append([img_dummy_label[y_label] for x in range(len(x_image[np.newaxis,:,:,:]))])
    x_image,y_image = np.concatenate(images),np.concatenate(images_label)
    return x_image,y_image

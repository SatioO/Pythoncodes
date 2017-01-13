# alexnet_helper.py

import os
import glob
from tqdm import tqdm
import random
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import cv2



def alex_agument_data(image,min_side=256,crop_size=224):
    """
    returns a numpy array of shape NHWC.

    Args:
    image: A numpy array of an image shape(HWC).
    """
    h = min_side=256
    l = int(image.shape[1]*h/image.shape[0])
    resize = cv2.resize(image,(l,h),interpolation=cv2.INTER_AREA)
    ## Take the 4 crops and center crop
    h1 = resize.shape[0]
    w1 = resize.shape[1]
    resize = resize[:,w1/2-min_side/2:w1/2+min_side/2,:]
    h1 = resize.shape[0]
    w1 = resize.shape[1]
    crop1 = resize[0:crop_size,0:crop_size,:]
    crop2 = resize[h1-crop_size:h1,0:crop_size,:]
    crop3 = resize[0:crop_size,w1-crop_size:w1,:]
    crop4 = resize[h1-crop_size:h1,w1-crop_size:w1,:]
    if w1 % 2 != 0: # to make sure that we deal with even number
        w1 = w1+1
    crop5 = resize[h1/2-crop_size/2:h1/2+crop_size/2,w1/2-crop_size/2:w1/2+crop_size/2,:]
    crops = np.concatenate((crop1[np.newaxis,:,:,:],crop2[np.newaxis,:,:,:],crop3[np.newaxis,:,:,:],crop4[np.newaxis,:,:,:],crop5[np.newaxis,:,:,:]))
    flip_h = [cv2.flip(crops[i],1) for i in range(len(crops))]
    images = np.concatenate((crops,flip_h))
    return images


def alex_dev_image_reader(filelist,img_dummy_label,size=(256,256),normalize = True):
    images = []
    images_label = []
    for j in range(len(filelist)):
        x_image,y_label= image_read(filelist[j])
        x_images = alex_agument_data(x_image)
        images.append(x_images)
        images_label.append([img_dummy_label[y_label] for x in range(len(x_images))])
        x_image,y_image = np.concatenate(images),np.concatenate(images_label)
        x_image,y_image = shuffle(x_image,y_image,random_state=0)
        if normalize == True:
            x_image = x_image/255.0
    return x_image,y_image


def alex_val_image_reader(image,size=(256,256),normalize = True,crop_size = 224):
    images = []
    x_image,y_label = image_read(image)
    x_images = alex_agument_data(x_image,size[0],crop_size)
    return x_images 

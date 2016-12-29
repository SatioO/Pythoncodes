"""
Problems with training the image
   - Images are of different shape . And not sure what the future Images are going to be
   - Some Images are taken in day and some in night


"""

import numpy as np
import pandas as pd
import glob,cv2
import random
from PIL import Image
import os
from sklearn.utils import shuffle
from tqdm import tqdm
from pre_process import *

import warnings
warnings.filterwarnings("ignore")

# create a dataframe of Image_labels along with Name
# labels = pd.DataFrame
# folderlist = glob.glob("/Users/Satish/Downloads/kaggle-fish/data/*")
#
# for i in range(len(folderlist)):
#     filelist = glob.glob(folderlist[i]+"/*")
#     for f in range(len(filelist)):
#         index = filelist[f].rsplit("/")[-2]+"_"+filelist[f].rsplit("/")[-1]
#         level = filelist[f].rsplit("/")[-2]
#         labels = labels.append({"index":index,"level":level}, ignore_index=True)
#
#
# labels.to_csv("labels.csv",sep=",")

"""
Plan
 - First need to find how to optimally resize all the Images to the same size , without loosing any information
            - Try using kernals
 - Try to fit the the above function in our Image generator function

"""

def filelist(folder):
    """
    Returns a dictionary with keys as folder names and value as a list of file locations

    Args:
    folder: string of folder name which contain all sub folder of images

    Returns:
    A dictionary object with keys - folder name and values - list of file locations of respective folder
    """
    file_dict={}
    folderlist = glob.glob(os.getcwd()+"/"+folder+"/*")
    for i in tqdm(folderlist):
        filelist = glob.glob(i+"/*")
        filename = i.rsplit("/")[-1]
        file_dict[filename]= filelist

    return file_dict


""" Function to split data into train and test """

def train_test(folderlist, test_size=0.3, random_state = 0):
    """
    Returns two dict objects.

    Args:
    folderlist: A dict which contains the label as key and its respective Images locations as a list in place of value.
    test_size: size of the test data required
    random_state: seed required

    Returns
    Splits folderlist into two dictionaries and returns them as train and test

    """
    random.seed(random_state)
    test = {}
    train = {}
    for i in folderlist.keys():
        test[i] = [random.choice(folderlist[i]) for x in range(int(len(folderlist[i])*test_size))]
        train[i] = [x for x in folderlist[i] if x not in test[i]]
    return train,test



def Dev_Image_data_generator(folderlist,resize = (920,1200),Transformation = True, scaling = True, batch_size = 16):
    """
    Yields a tuple (x,y) with x - batch of images(numpy array), y - image label (numpy array)

    Args:
    folderlist : A dictionary object
    resize : tuple of (x,y)
    Transformation : If True Data Aguementation is done
    scaling : If True , every Image is scaled
    batch_size : The batch_size to yield for every iteration

    returns:
    A tuple with Images and labels as numpy arrays.
    """

    while True:
        total_classes = len(folderlist.keys())
        keys = folderlist.keys()
        Images = []
        Image_label = []
        for key in folderlist.keys():
            img_label = random.choice(folderlist[key])
            img = Image.open(img_label,'r')
            h = resize[1]
            l = int(img.size[1]*h/img.size[0])
            img = img.resize((h,l), Image.ANTIALIAS)
            background = Image.new('RGB', (resize[1], resize[0]), (255, 255, 255))
            img_w, img_h = img.size
            bg_w, bg_h = background.size
            offset = (int((bg_w - img_w) / 2), int((bg_h - img_h) / 2))
            background.paste(img, offset)
            background = np.asarray(background)
            if Transformation == True:
                rotation = rotate(background,random.choice(range(360)))
                translate = translate_xy(background,random.choice(range(resize[0]/4)),random.choice(range(resize[1]/4)))
                flip = cv2.flip(rotation,1)
                Y = np.concatenate((rotation[np.newaxis,:,:,:],flip[np.newaxis,:,:,:],translate[np.newaxis,:,:,:]))
                Images.append(Y)
                Images.append(background[np.newaxis,:,:,:])
                Image_label.append([key for i in range(4)])
            else:
                Images.append(background[np.newaxis,:,:,:])
                Image_label.append([key])
        Image_label = np.concatenate(Image_label)
        Images = np.concatenate(Images)
        Image_label = np.array(pd.get_dummies(Image_label))
        X_Image , Y_Image = shuffle(Images,Image_label,random_state=0)
        if scaling == True:
            X_Image = X_Image/255
        else:
            X_Image = X_Image
        batches = int(len(X_Image)/batch_size)
        for batch in range(batches):
            x = X_Image[batch*batch_size:(batch+1)*batch_size,:,:,:]
            y = Y_Image[batch*batch_size:(batch+1)*batch_size]
            yield((x,y))


def Valid_Image_data_generator(folderlist,resize = (920,1200),Transformation = True, scaling = True):
    """
    Yields a tuple (x,y) with x - batch of images(numpy array), y - image label (numpy array)

    Args:
    folderlist : A dictionary object
    resize : tuple of (x,y)
    Transformation : If True Data Aguementation is performed.
    scaling : If True , every Image is scaled by 255
    batch_size : The batch_size to yield for every iteration

    returns:
    A tuple with Images and labels as numpy arrays.
    """

    while True:
        total_classes = len(folderlist.keys())
        keys = folderlist.keys()
        Images = []
        Image_label = []
        for key in tqdm(folderlist.keys()):
            for j in range(len(folderlist[key])):
                img_label = folderlist[key][j]
                img = Image.open(img_label,'r')
                h = resize[1]
                l = int(img.size[1]*h/img.size[0])
                img = img.resize((h,l), Image.ANTIALIAS)
                background = Image.new('RGB', (resize[1], resize[0]), (255, 255, 255))
                img_w, img_h = img.size
                bg_w, bg_h = background.size
                offset = (int((bg_w - img_w) / 2), int((bg_h - img_h) / 2))
                background.paste(img, offset)
                background = np.asarray(background)
                if Transformation == True:
                    rotation = rotate(background,random.choice(range(360)))
                    translate = translate_xy(background,random.choice(range(resize[0]/4)),random.choice(range(resize[1]/4)))
                    flip = cv2.flip(rotation,1)
                    Y = np.concatenate((rotation[np.newaxis,:,:,:],flip[np.newaxis,:,:,:],translate[np.newaxis,:,:,:]))
                    Images.append(Y)
                    Images.append(background[np.newaxis,:,:,:])
                    Image_label.append([key for i in range(4)]) # Four because we are doing rot,trans,flip and one original Image
                else:
                    Images.append(background[np.newaxis,:,:,:])
                    Image_label.append([key])
        Image_label = np.concatenate(Image_label)
        Images = np.concatenate(Images)
        Image_label = np.array(pd.get_dummies(Image_label))
        X_Image , Y_Image = shuffle(Images,Image_label,random_state=0)
        if scaling == True:
            X_Image = X_Image/255
        else:
            X_Image = X_Image
        return (X_Image,Y_Image)

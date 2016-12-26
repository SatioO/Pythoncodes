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

import warnings
warnings.filterwarnings("ignore")

# create a dataframe of Image_labels along with Name
labels = pd.DataFrame
folderlist = glob.glob("/Users/Satish/Downloads/kaggle-fish/data/*")

for i in range(len(folderlist)):
    filelist = glob.glob(folderlist[i]+"/*")
    for f in range(len(filelist)):
        index = filelist[f].rsplit("/")[-2]+"_"+filelist[f].rsplit("/")[-1]
        level = filelist[f].rsplit("/")[-2]
        labels = labels.append({"index":index,"level":level}, ignore_index=True)


labels.to_csv("labels.csv",sep=",")

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



def Image_data_generator(folderlist,labels,resize = (920,1200),Transformation = True, scaling = True, batch_size = 16):
    """
    Yields a tuple (x,y) with x - batch of images(numpy array), y - image label (numpy array)

    Args:
    folderlist : A dictionary object
    labels : pandas dataframe with index as labels name
    resize : tuple of (x,y)
    Transformation : If True Data Aguementation is done
    scaling : If True , every Image is scaled
    batch_size : The batch_size to yield for every iteration

    returns:
    A tuple with Images and labels as numpy arrays.
    """

    while True:
        total_classes = len(folderlist.keys())
        keys = folderlist.keys[]
        Images = []
        Image_label = []
        for key in foldelist.keys():
            img_label = random.choice(folderlist[key])
            img = Image.open(img_label,'r')
            h = resize[1]
            l = int(img.size[1]*h/img.size[0])
            img = img.resize((h,l), Image.ANTIALIAS)
            background = Image.new('RGB', (resize[1], resize[0]), (255, 255, 255))
            img_w, img_h = img.size
            bg_w, bg_h = background.size
            offset = ((bg_w - img_w) / 2, (bg_h - img_h) / 2)
            background.paste(img, offset)
            Images.append(background)
            if Transformation == True:
                rotation = np.array([rotate(background[j],random.choice(range(360))) for j in range(1)])
                translate = translate_xy(background,random.choice(range(resize[0]/4)),random.choice(range(resize[1]/4)))
                flip = np.array([cv2.flip(rotation[j],1) for j in range(len(rotation))])
                Y = np.concatenate((rotation,flip,translate))
                Images.append(Y)
                Images.append(background[np.newaxis,:,:,:])
                Image_label.append([key for i in range(len(Images))])
        Image_label = np.concatenate(Image_label)
        Images = np.concatenate(Images)
        Image_label = np.array(pd.get_dummies(Image_label))
        X_Image,Y_Image = shuffle(Images,Image_label,random_state=0)
        if scaling == True:
            X_Image = X_Image/255
        else:
            X_Image = X_Image
        batches = int(len(X_Image)/batch_size)
        for batch in range(batches):
            x = X_Image[batch*batch_size:(batch+1)*batch_size,:,:,:]
            y = Y_Image[batch*batch_size:(batch+1)*batch_size]
            yield((x,y))

""" helper functions """

import cv2
import glob
import numpy as np


def img_location_reader(foldername):
    """
    Returns a list of lists with each list containing the locations of each class
    Args:
    foldername : string of the foldername where the images reside
    """
    folderlist = glob.glob(os.getcwd()+"/"+foldername+"/*")
    imglist = [glob.glob(i+"/*") for i in folderlist]
    return imglist


def image_read(image_location):
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
    y = np.zeros([36,36,3])
    y[18-14:18+14,18-14:18+14,:]=x
    return y

# Libaries

from PIL import Image,ImageChops
from PIL import ImageFile

import glob # for creating a list of filenames from a directory


from skimage import io
import numpy as np 
import pandas as pd 

from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Convolution2D, MaxPooling2D



"""
pre-processing Images
Transformations need to do on Images  - please check the link below to see the output of each function

"""


## First Translation
import numpy as np
import cv2

# Translate
def translate(image,x,y):
    """
    Return a translated Image of the input Image
    Args:
    image : Input Image you want to translate (shape height * width * channel)
    x,y : -ve x _ left, +ve x _ right, -ve y _ up , +ve y down

    Return :
    Translated Image

    """
    M = np.float32([[1, 0, x],[0, 1, y]])
    shifted = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
    return shifted


# rotate
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


# read Image RGB mode
def Image_read(image):
    """
    A function to convert the mode of GBR to RGB mode
    Args:
    image = input the Image location.
    Returns:
    return a numpy array in RGB mode
    """
    x = cv2.imread(image)
    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    return x


def day_to_night(image):
    arr = img *np.array([0.1,0.2,1.7])
    img = (255*arr/arr.max()).astype(np.int8)
    return img


# Shear range and zoom range
def shear_image(image,shear = 0.2):
    from skimage import transform
    afine = transform.AffineTransform(shear = shear)
    modified = tf.warp(image,afine)
    return modified



 # Brightness or adding colour to Dark Image. Generally green light
def bright_image(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h +=60
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2RGB)
    return image

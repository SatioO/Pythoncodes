""" Kernels
 - Blurring
 - sharpening
 - edge detection etc

 """
import argparse 
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required= True, help = "path to the image")
args = vars(ap.parse_args())

# load the image and convert it to grayscale 
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Original",image)

 
""" # erosions 
A foreground pixel in the input image will be kept only if all pixels inside the 
structuring element are >0. otherwise, the pixels are set to 0(i.e. background)

- useful in removing small blobs in an image or disconnecting two connected images 
"""
for i in range(0,3):
 eroded = cv2.erode(gray.copy(),None,iterations=i+1)
 cv2.imshow("eroded",eroded)

#$ python morphological.py --image pyimagesearch_logo.png

""" # Dilations 
- The opposite of erosion is a dilation.
- Dilation increase the size of foreground object and are especially useful for joining broken parts od
an image together.
- Just as an erosion, also utilize structuring elements- a center pixel p of the structuring 
element is set to white if Any pixel in the structuring element is >0
"""
for i in range(0,3):
 dilated = cv2.dilate(gray.copy(), None, iterations= i+1)
 cv2.imshow("dilated",dilated)

""" # Opening 
- An opening is an erosion followed by a dilation 
- Performing an opening operation allows us to remove small blobs from an image:
   first an erosion is applied to remove the small blobs , then a dilation is applied to 
   regrow the size of the original object
"""
kernelSizes = [(3,3),(5,5),(7,7)]

for kernelSize in kernelSizes:
 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize) #rectangle structuring element
 # cv2.MORPH_CROSS - cross shape structuring element
 # cv2.MORPH_ELLIPSE - circular structuring element
	opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	cv2.imshow("Opening: ({}, {})".format(kernelSize[0], kernelSize[1]), opening)

""" # Closing 
- The exact opposite of opening 
- A closing is a dilation followed by an erosion
"""

for kernelSize in kernelSizes:
 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
 closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
 cv2.imshow("Closing", closing)

""" # Morphological Gradient
- It is the difference between the dilation and erosion
- It is useful for determining the outline of a particular object of an image
"""

# loop over the kernels and apply a "morphological gradient" operation
# to the image
for kernelSize in kernelSizes:
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
	cv2.imshow("Gradient: ({}, {})".format(kernelSize[0], kernelSize[1]), gradient)
 
""" # Top hat/ white hat 
- A top hat morphological operation is the difference between the original input image
and the opening 
- It is used to reveal bright regions of an image on dark backgrounds
"""
 
# construct a rectangular kernel and apply a blackhat operation which
# enables us to find dark regions on a light background
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
 
# similarly, a tophat (also called a "whitehat") operation will enable
# us to find light regions on a dark background
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

# Blackhat operation is the difference between the closing of the input image and the input image itself

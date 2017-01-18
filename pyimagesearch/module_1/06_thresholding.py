""" # Thresholding
- It is the binarization of an image 
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

# apply the blurring
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.gaussianBlur(gray,(7,7),0)

# apply basic thresholding -- the first parameter is the image we want to threshold, the second value is our threshold check
# if a pixel value is greater than our threshold (in this case,200), we set it to be BLACK, otherwise it is WHITE.
(T, threshInv) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Binary Inverse", threshInv)

# using normal thresholding (rather than inverse thresholding), we can change the last argument in the function to make the coins
# black rather than white.
(T, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Binary", thresh)

# apply Otsu's automatic thresholding -- Otsu's method automatically determines the
# best threshold value 'T' for us
(T, threshInv) = cv2.threshold(blurred,0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Threshold", threshInv)

#- Assumes a bi-modal distribution of the grayscale pixel intensities of our input image
#- when lightning conditions are non uniform , it is a serious problem

# Adaptive thresholding - local thresholding
# T = mean(Il) - C
# instead of manually specifying the threshold value, we can use adaptive
# thresholding to examine neighborhoods of pixels and adaptively threshold
# each neighborhood -- in this example, we'll calculate the mean value
# of the neighborhood area of 25 pixels and threshold based on that value;
# finally, our constant C is subtracted from the mean calculation (in this
# case 15)
thresh = cv2.adaptiveThreshold(blurred, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15) #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
cv2.imshow("OpenCV Mean Thresh", thresh)

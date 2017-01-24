"""

- Moment-baes approaches in imaging
- Complex zernike moments
- Moments in Image Processing

pip install mahotas

import mahotas
moments = mahotas.features.zernike_moments(image,21, degree = 8)
"""


from scipy.spatial import distance as dist
import numpy as np
import mahotas
import cv2, glob, os

def descibe_shapes(image):
    #initialize the list of shape features
    shapeFeatures = []

    #convert the image to grayscale , blur it , and threshold it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]

    #perform a series of dilations and erosions to close holes in the shapes
    thresh = cv2.dilate(thresh, None, iterations =4)
    thresh = cv2.dilate(thresh, None, iterations = 2)

    # detect the contours in the edge map
    (cnts,_)= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #loop over the contours
    for c in cnts:
        mask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # extract the bounding box ROI from the mask
        (x, y, w, h) = cv2.boundingRect(c)
        roi = mask[y:y+h,x:x+w]

        features = mahotas.features.zernike_moments(roi,cv2.minEnclosingCircle(c)[1], degree = 8)
        shapeFeatures.append(features)

    return (cnts, shapeFeatures)


# load the reference image containing the object we want to detect, then describe the game region
refImage = cv2.imread("")
(_, gameFeatures) = describe_shapes(refImage)

#load the shapes image, then describe each of the images in the image
shapeImage = cv2.imread("shapes.png")
(cnts, shapeFeatures) = describe_shapes(shapesImage)

#compute the Euclidean distances between the video game
D = dist.dist(gameFeatures, shapeFeatures)
i = np.argmin(D)


for (j, c) in enumerate(cnts):
    if i != j:
        box = cv2.minAreaRect(c)
        box = np.int0(cv2.cv.BoxPoints(box))
        cv2.drawContours(shapesImage, [box], -1, (0, 0, 255), 2)

box = cv2.minAreaRect(cnts[i])
box = np.int0(cv2.cv.BoxPoints(box))
cv2.drawContours(shapesImage, [box], -1, (0, 255, 0), 2)
(x, y, w, h) = cv2.boundingRect(cnts[i])
cv2.putText(shapesImage, "FOUND!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

# show the output images
cv2.imshow("Input Image", refImage)
cv2.imshow("Detected Shapes", shapesImage)
cv2.waitKey(0)


"""
- To describe multiple shapes in an image, be sure to extract the ROI of each object, and then extract zernike Moments from each ROI
- Pay attention to radius and degree parameters

Pros:
1. Very fast to compute
2. Low dimensional
3. Very good at describing simple shapes
4. Fairly simple to tune the radius and degree parameters

Cons:
2. Normally used for simple 2D shape - as shapes become more complex, Zernike Moments doesnot perform well
3. Just like Hu Moments, Zernike Moments calculations are based on the initial centroid computation - if the initial centroid cannot be repeated for similar shapes, then Zernike Moments will not obtain good matching accuracy.

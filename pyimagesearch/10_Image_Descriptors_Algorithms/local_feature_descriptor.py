"""
- obtain highlevel understanding of what local features are
- Understanding the differences between local features and global features such as HoG, LBP etc
- Define terms such as keypoint detector and feature extractor
- Understand some of the challenges associated with local features

1) Flat Regions: Low density patches don't make good features.
2) Edge regions are more interesting and discriminative than flat regions.
3) corners are considered to be very good interesting and discriminative regions to detect.

Keypoint Detection and feature extraction:
The process of finding and describing interesting regions of an image is broken down into two phases: keypoint detection and feature extraction.
"""

#FAST
# The fast keypoint detector is used to detect corners in Images. It is primarily suited for real-time or resource constrained            # applications where keypoint can quickly be computed.

import numpy as np
import cv2

image = cv2.imread("next.png")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detect FAST keypoints in the image
detector = cv2.FastFeatureDetector_create()
kps = detector.detect(gray)
print (len(kps))


#loop over the key points and draw them
for kp in kps:
    r = int(0.5*kp.size)
    (x,y) = np.int0(kp.pt)
    cv2.circle(image,(x,y),r,(0,255,255),2)



"""
#HARRIS
# combined corner and edge detector


compute G_x and G_y
M = [[Gx^2, GxGy],[GxGy, Gy^2]]
R = det(M) - k*(trace(M))^2

det(M) = \lamba_1 * \lamba_2
trace(M) = \lamba_1 + \lamba_2

R = det(M) - k*(trace(M))^2

- |R| is small, then we are examining a "flat" region of the image. Thus the region is not a keypoint
- R < 0, which happens when \lamba_1 >> \lamba_2 or \lamba_2 >> \lamba_1, then the region is a edge. Again the region is not a keypoint
- The only time the region can be considered a keypoint is when both |R| is large, which corresponds to \lamba_1 and when \lamba_2 being approximately equal. if this holds , then the region is indeed a keypoint.

"""
detector = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]



"""
#GFTT - Shi-Tomasi keypoint detector

R = \lamba_1 * \lamba_2 - k(\lamba_1+\lamba_2)^2

R = min(\lamba_1,\lamba_2)
if R > T (Threshold), then mark region as a corner

- \lamba_1 and \lamba_2 are < T ; thus the region is not a key point
- \lamba_1 < T, so we cannot mark the region as keypoint
- \lamba_2 < T, cannot mark the region as keypoint
- \lamba_2 and \lamba_1 > T, we can mark the region as a corner keypoint in Region
"""

"""
DoG - Difference of Gaussian keypoint detector
- scale space images : Take the original image and create progressively blurred (using Guassian kernel) versions of it. We then half the size of the image and repeat. Images of the same size are called Octaves.
- Difference of Guassian : We take two consequtive images in the octave and subtract them from each other. We then move to the next two consequtive images in the octave and repeat the process.
- Finding local maxima and local minima : The pixel can be considered a "key point" if the pixel intensity is larger or smaller than all of its 8 surrounding regions | Above and below layer (9 pixels each)+ image layer(8 pixels each) = 26 . if the pixel X is greater than or less than all 26 of its neighbours, then it is a key point.
"""


"""
Fast Hessian - used with SURF descriptor
- Invented to speed up GFTT
- sets of "box filters" and convolve them with the image in an attempt to approximate the Difference of Guassians
-  contruct Hessian matrix
- A region is then marked as a "key point" if the candidate score is greater than 3*3*3 neighbour.
"""

"""
STAR - approximate the Difference of Guassians for increased speed
"""

"""
MSER -
Blob is defined by areas of an image that exhibit
1) connected components
2) near uniform pixel intensities
3) contrasting background

- step1 = for each of the Threshold images , perfom a connected component analysis on the binary regions.
- step2 = Compute the Area A(i) of each of these connected components
- step3 = Monitor the Area A(i) of these connected components over multiple Threshold values. If the area remains relatively constant in Size, then mark the region as a keypoint

Use when regions you want to detect are
1) small.
2) relatively same pixel intensity.
3) surrounding by contrasting pixels.
"""


"""
Dense:
Mark every K pixel in the image as a keypoint.
"""

"""
BRISK:
- The original implementation of FAST by rosten and Drummond(2005) only examined the image at a single scale, meaning that it was unlikely keypoints could be repeated over multiple scales of the Image.
- Brisk is able to address this limitation by creating scale space images, similar to DoG. The general process is to half the size of the image for each layer of the pyramid.
- for each layer of the pyramid, the standard fast key point detector runs, with maximum response(the regions most likely to be a corner are taken across all levels of the image pyramid)
"""

"""
ORB:
just like Brisk, ORB is also an extension to the FAST keypoint detector. It, too , utilizes an image pyramid to account for multi-scale
keypoints; However, (and unlike Brisk), ORB adds rotational invariance as well.
- Compute key points like Brisk
- Take the top (rank) 500 points, using Harris keypoint detector
- rotational invariance is added in the third step
"""

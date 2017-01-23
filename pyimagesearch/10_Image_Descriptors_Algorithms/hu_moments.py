"""
- Hu proposed seven moments that can be used to characterize the shape of an object in an image
- Hu further demonstrates that these moments are invariants to changes in rotation, translation, scale, and reflection(i.e. mirroring)
- Moment based calculations are often sensitive to the initial calculation of the centroid.

- Hu moments should not be used where there is
   - noise
   - occlusion
   - lack of clear segmentation

standard moments - cv2.moment
Hu Moments - cv2.HuMoments

Moments
- mean
- variance - sd
- skew
- kurtosis

M(i,j) = \sum_{x}\sum_{y} x_i * y_i * I(x,y)
I(x,y) is the pixel intensity

centoid (x,y) = (M_{10}/M_{00}, M_{01}/M_{00})

M_{1} = (\mu_{20} + \mu_{02})
M_{2} = (\mu_{20} - \mu_{02})^{2} + 4\mu_{11}^{2})
M_{3} = (\mu_{30} - 3\mu_{12})^{2} + (3\mu_{21} - \mu_{30})^{2}
M_{4} = (\mu_{30} + \mu_{12})^{2} + (\mu_{21} + \mu_{03})^{2}
M_{5} = (\mu_{30} - 3\mu_{12})(\mu_{30} + \mu_{12})((\mu_{30} + \mu_{12})^{2} - 3*\mu_{21} + \mu_{03})^{2}) + (3\mu_{21} - \mu_{03})(\mu_{21} + \mu_{03})(3*\mu_{30} + \mu_{12})^{2} - (\mu_{21} + \mu_{03})^2)
M_{6} = (\mu_{20} - \mu_{02})((\mu_{30} + \mu_{12})^{2} - (\mu_{21} + \mu_{03})^{2}) + 4\mu_{11}(\mu_{30} + 3\mu_{12})(\mu_{21} + \mu_{03})
M_{7} = (3\mu_{21} - \mu_{03})(\mu_{30} + \mu_{12})((\mu_{30} + \mu_{12})^{2} - 3*\mu_{21} + \mu_{03})^{2}) - (\mu_{30} - 3\mu_{12})(\mu_{21} + \mu_{03})(3*\mu_{30} + \mu_{12})^{2} - (\mu_{21} + \mu_{03})^{2})
These seven moments then form our feature vector:

V = [M_{1}, M_{2},..., M_{7}]

"""

import cv2

image = cv2.imread("planes.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute the Hu Moments feature vector for the entire image and show it
moments = cv2.HuMoments(cv2.moments(image)).flatten()
print ("ORIGINAL MOMENTS:{}".format(moments))
cv2.imshow("Image", image)

# To correctly compute Hu Moments for each of the three aircraft silhouettes, we'll need to find the contours of each airplance, extract ROI surrounding the airplane , and then compute Hu Moments for each ROIs individually
(cnts,_) = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# loop over each contour
for (i,c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = image[y:y+h, x:x+w]
    moments = cv2.HuMoments(cv2.moments(roi)).flatten()


""" SHAPE OUTLIER DETECTION """
"""
Pros :
1. Very fast to compute
2. Low dimensional
3. Good at describing simple shapes
4. No parameters to tune
5. Invariant to changes in rotation, reflection, and scale
6. translation invariance is obtained by using a tight cropping of the object to be descibed

cons :
1. Requires a very precise segmentation of the object to be described, which is often hard in the real world
2. Normally used for simple 2D shape - as shapes become more complex, Hu Moments are not often used
3. Hu Moment calculations are based on the initial centroid computation - if the initial centroid cannot be repeated for similar shapes, then Hu Moments will not obtain good matching accuracy.

"""

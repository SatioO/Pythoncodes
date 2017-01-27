# Gradients
"""
- Histogram of oriented Gradients and SIFT are built upon image gradient representation

- Gradient
- Gradient magnitude  - How strong the change in image intensity is
- Gradient orientation - in which direction the change in intensity is pointing
- Sobel and Scharr kernels

Gx = [[-1 0 1],
      [-2 0 2],
      [-1 0 1]]

Gy = [[-1 -2 -1],
      [0   0  0],
      [1   2  1]]
Application - Edge detection
"""

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#compute the gradients along with X and Y axis , respectively
gX = cv2.Sobel(gray, ddepth= cv2.CV_64F,dx = 1 , dy=0) # cv2.Scharr and use ksize=-1 for optimal parameter value
gY = cv2.Sobel(gray, ddepth= cv2.CV_64F,dx = 0, dy=1)

# gX and gY images are floating point type, convert to unsigned 8-bit
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)

# combine the sobel X and Y representation into a single image
sobelCombined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)


# compute the gradient magnitude and orientation respectively
mag = np.sqrt((gX**2)+(gY**2))
orientation = np.arctan2(gY,gX)*(180/np.pi) % 180

# find all the pixels that are within the upper and low angle boundaries
idxs = np.where(orientation >= args["lower_angle"], orientation, -1)
idxs = np.where(orientation >= args["upper_angle"], orientation, -1)
mask = np.zeros(gray.shape, dtype = "uint8")
mask[idxs > -1] = 255

# Edge detector
# Canny edge detector is arguably the most well known and the most used edge detector in all the computer-vision and image processing
# - John F.Canny in his 1986 Paper . A computational Approach to Edge Detection
"""
edge - A sharp difference and change in pixel values | discontinuities in pixel intensity
- Step edge __|--
- Ramp edge __/---
- Ridge edge _/--\_
- Roof edge __/\__

Process -
Apply Guassian smooting to reduce the noise
compute gX and gY using sobel kernel
Apply non-maxima supression to keep only the local maxima of gradient magnitude pixels that are pointing in the direction of the gradient - edge thining process
defining and applying Tupper and Tlower thresholds for Hysteresis thresholding

"""
blurred = cv2.GuassianBlur(gray, (5,5), 0)
wide = cv2.Canny(blurred,10,100)
mid = cv2.Canny(blurred, 30, 150)
tight = cv2.Canny(blurred, 240, 250)

# check for auto canny in imutils


""" 
- Understand the role kernels play in smoothing and blurring 
- Apply simple average blurring (averaging)
- Apply weighted Gaussian blurring (Gaussian blurring)
- Understand the importance of median filter (median filtering)
- Utilize bilateral filtering to blur an image while preserving edges (bilateral filtering)

"""

""" # Averaging 
simple average of kernal size pixels 
- kernel of size (3*3)
  k = 1/9 *[[1,1,1],
            [1,1,1],
            [1,1,1]]
- kernel of size (5*5)
  k = 1/25 *[[1,1,1,1,1],
             [1,1,1,1,1],
             [1,1,1,1,1],
             [1,1,1,1,1],
             [1,1,1,1,1]]
 - The larger your smooting kernel is , the more blurred your image will look
"""

kernelSizes=[(3,3), (9,9), (15,15)]
#loop over the kernel sizes and apply an "average" blur to the image

for (kX, kY) in kernelSizes:
  blurred = cv2.blur(image,(kX, kY))
  cv2.imshow("blurred ({},{})".format(kX,kY),blurred)
  
""" # Gaussian Blurring 
we use weighted mean , where neighborhood pixels that are closer to the central
pixel contribute more "weight" to the Average.
- Gaussian smoothing is used to remove noise that approximately follows a 
Gaussian distribution
"""
for (kX, kY) in kernelSizes:
	blurred = cv2.GaussianBlur(image, (kX, kY), 0)
	cv2.imshow("Gaussian ({}, {})".format(kX, kY), blurred)

""" # Median 
- removing salt and pepper noise 
"""
for k in (3, 9, 15):
	blurred = cv2.medianBlur(image, k)
	cv2.imshow("Median {}".format(k), blurred)
  
""" # Bilateral
- reduce noise while still maintaining edges, we can use bilateral blurring 
  - First Guassian - considers spatial neighbours 
  - Second Guassian - models the pixel intensity of the neighborhood.
"""
params = [(11, 21, 7), (11, 41, 21), (11, 61, 39)]
 
# loop over the diameter, sigma color, and sigma space
for (diameter, sigmaColor, sigmaSpace) in params:
	# apply bilateral filtering and display the image
	blurred = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
	title = "Blurred d={}, sc={}, ss={}".format(diameter, sigmaColor, sigmaSpace)
	cv2.imshow(title, blurred)




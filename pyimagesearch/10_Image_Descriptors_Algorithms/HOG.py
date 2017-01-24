"""
HOG

- Normalizing the image prior to description
- Computing gradients in both x and y directions
- Obtaining weighted votes in spatial and orientation cells
- Contrast normalizing overlapping spatial cells.
- Collecting all HOG to form the final feature vector

Imp parameters
- orientations
- pixels_per_cell
- cells_per_block

HOG is used to describe the structural shape and appearance of an object in an image, making them excellent descriptors for object classification.

steps
1. normalizing the image prior to description
2. Gradient computation
3. Weighted votes in each cell
4. Contrast normalization over blocks

http://crcv.ucf.edu/courses/CAP5415/Fall2013/Lecture-5.5-HOG.pdf

Feature Vector length:
- Image size      =          128*128         | 32*32
- Pixels_per_cell =              4*4         | 4*4
- Cells per block =              2*2         | 2*2
- cells block     =(128*128)/(4*4) =(32*32)  | (32*32)/(4*4) =(8*8)
- orientations    =              9           |     9
- As the block moves we get = 2*2*9 = 36     | 2*2*9 = 36
- total length = (32-1)*(32-1)*36 = 34,596   | (8-1)*(8-1)*36 = 1764
"""

from skimage import feature
H = feature.hog(logo, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True)
# if transform_sqrt doesnt work use noramlise

#visualizing hog features
from skimage import exposure
(H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), transform_sqrt=True, visualise = True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")

cv2.imshow("HOG Image", hogImage)



from sklearn.neighbours import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import mahotas, glob, cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required = True, help = "Path to the dataset of textures")
ap.add_argument("-t", "--test", required = True, help = "Path to the test images")
args = vars(ap.parse_args())

# initialize the data matrix and list of labels
print ("INFO etracting  features ...")
data = []
labels = []


for imagePath in path.list_images(args["training"]):
    make = imagePath.split("/")[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    (_,cnts,_) = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key = cv2.contourArea)

    # extract the logo of the car and resize it to a canonical width and height
    (x, y, w, h) = cv2.boundingRect(c)
    logo = gray[y:y+h, x:x+w]
    logo = cv2.resize(logo, (200, 100))

    H = feature.hog(logo, orientations=9, pixels_per_cell=(10,10), cells_per_block=(2,2), transform_sqrt=True)

    data.append(H)
    labels.append(make)


print ("[INFO] training classifier")
model = KNeighborsClassifier(n_neighbors = 1)
model.fit(data, labels)


### Test on the test images . write your own code.


"""
Pros:
- Very powerful descriptor
- Excellent at representing local appearance
- Extremely useful for representing structural objects that do not demonstrate substantial variation in form
- very accuarate for object classification


cons:
- can result in very large feature vector , leading to large storage costs and computationaly exprensive feature vector comparision
- often non-trival to tune the orientations, pixel-per-cell, and cells_per_block parameters
- Not the slowest but nowhere near the fastest
- if the object to be described exhibits substantial structural variation, then the standard vanilla implementation of HOG will not perform well.

"""

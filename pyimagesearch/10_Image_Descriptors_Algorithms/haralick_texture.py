"""
Haralick texture features are used to describe the "texture" of an Image . Quantifying and representing the
1) Feel
2) Appearance
3) consistency of a surface


Haralick features are derived from the gray level Co-occurence Matrix (GLCM). This matrix records how many times two gray-level pixels adjacent to each other appear in an image. Then based on this matrix, Haralick proposes 13 values that are extracted from the GLCM to quantify texture.

Exp : Determining a road is pavel vs. gravel

we calculate how many times adjacent pixels came together and impute them in the matrix . This is not restricted to left-right,
1. left to right
2. top to bottom
3. top-left to bottom-right
4. top-right to bottom-left

4 GLCM matrices to compute our Haralick features for each of GLCM
these values are simple statistics computed from the GLCM used to characterize and represnt
1) contrast
2) correlation
3) dissimilarity
4) entropy
5) homogeneity

import mahotas
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
features = mahotas.features.haralick(gray).mean(axis=0)
"""

from sklearn.svm import LinearSVC
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

# loop over the dataset of images
for imagePath in glob.glob(""):
    image  = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture = imagePath.rsplit("/")[0]

    # extract Haralick texture features in 4 directions, then take the mean of each direction
    features = mahotas.features.haralick(image).mean(axis=0)

    data.append(features)
    labels.append(texture)

model = LinearSVC(C = 10.0, random_state = 42)
model.fit(data, labels)


# loop over the test images
for imagePath in glob.glob("test*"):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = mahotas.features.haralick(gray).mean(axis=0)

    # classify the test image
    pred = model.predict(features.reshape(1, -1))[0]
    cv2.putText(image, pred, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    cv2.imshow("Image", image)
    cv2.waitKey(0)

"""
Pros:
- Very fast to compute
- Low dimensionality
- No parameters to tune

Cons:
- Not very robust against changes in rotation.
- Very sensitive to noise - small perturbations in the grayscale image can dramatically affect the construction of the GLCM.
- Similar to Hu Moments, basic statistics are often not discriminative enough to distinguish between many different kinds of texture

"""

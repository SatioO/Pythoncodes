"""
SIFT
- Undestand how SIFT works.
- Use opencv to extract SIFT feature vectors from an image


Requires a set of Input keypoints
  - from each keypoint, take 16*16 pixel region surrounding the centre pixel of the keypoint region
      - Divide 16*16 pixel region into 16 4*4 pixel windows
          - For each of the 16 windows, compute
               - Gradient magnitude and orientation
                    - Given both , we contruct 8 bin histograms for each of the (4*4) pixel windows
                    - The amount added to each bin is dependent on the magnitude of the gradient
                    - utilize Guassian weighting (the futher the pixel is from the keypoint center, the less it contributes to the overall histogram)
                    - Collect all the 16 of these 8-bin orientation and concatenate them together
        Feature vector = 16*8 = 128 dim
        we end up L2 normalizing the entire feature vector. At this point , our SIFT feature vector is finished and ready to be compared to other SIFT feature vectors

        - N feature vectors per Image, where N is the number of detected keypoints
        - N detected points , we get N * 128 feature vectors after applying SIFT

"""

import argparse, cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# initalize the keypoint detector and local invariant descriptor
detector = cv2.FeatureDetector_create("SIFT")
extractor = cv2.DescriptorExtractor_create("SIFT")

image = cv2.imread(arg["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kps = detector.detect(gray)
(kps, descs) = extractor.compute(gray, kps)


"""
RootSIFT

sits on the top of original SIFT implementation and doesnot require any changes to the original SIFT source code.

step-1 : compute SIFT descriptors using your favourite SIFT library
step-2 : L1-normalize each SIFT vector
step-3 : Take the square root of each element of SIFT vector

"""

import cv2
import numpy as np

class RootSIFT:
    def __init__(self):
        self.extractor = cv2.DescriptorExtractor_create("SIFT")

    def compute(self, image, kps, eps=1e-7):
        (kps, descs) = self.extractor.compute(image, kps)

        if len(kps) == 0:
            return ([], None)

        # apply the hellinger kernal by first L1-normalizing and taking the square root
        descs /= (descs.sum(axis=1, keepdims = True) + eps)

        return (kps, descs)


"""
SURF

Advantages over SIFT:
1. Faster
2. only half the size of SIFT descriptors

step1 : Apply Fast Hessian and detect keypoints
step2 : loop over each keypoint and extract 4*4 sub area
step3 : For each of these 4*4 sub-areas, Haar Wavelet responses are extracted at 5*5 regularly-shaped sample points
step4 : For each of these 4*4 sub-areas, we compute sigma(dx) sigma(dy) sigma(|dx|) sigma(|y|)
         4*4*4 = 64 dims

"""
detector = cv2.FeatureDetector_create("SURF")
extractor = cv2.DescriptorExtractor_create("SURF")

image = cv2.imread(arg["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kps = detector.detect(gray)
(kps, descs) = extractor.compute(gray, kps)


"""
Real - valued feature Extraction and Matching

- Extract keypoints and local invariant descriptors from two images that contain the same object, but are captured using different camera sensors and under varying lightning conditions and viewing angles
- Apply feature matching to match the keypoints and descriptors from image #1 to image #2.
- Learn about David Lowe's ratio test for efficient feature matching

"""
# import the necessary packages
from __future__ import print_function
from pyimagesearch.descriptors import RootSIFT
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="Path to first image")
ap.add_argument("-s", "--second", required=True, help="Path to second image")
ap.add_argument("-d", "--detector", type=str, default="SURF",
  help="Keypoint detector to use")
ap.add_argument("-e", "--extractor", type=str, default="SIFT",
  help="Feature extractor to use")
ap.add_argument("-m", "--matcher", type=str, default="BruteForce",
  help="Feature matcher to use")
ap.add_argument("-v", "--visualize-each", type=int, default=-1,
  help="Whether or not each match should be visualized individually")
args = vars(ap.parse_args())



detector = cv2.FeatureDetector_create(args["detector"])
extractor = cv2.DescriptorExtractor_create(args["matcher"])

if args["extractor"] == "RootSIFT":
    extractor = RootSIFT()

else:
    extractor = cv2.DescriptorExtractor_create(args["extractor"])


imageA = cv2.imread(arg["first"])
imageB = cv2.imread(arg["second"])
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

kpsA = detector.detect(grayA)
kpsB = detector.detect(grayB)

(kpsA, featuresA) = extractor.compute(grayA, kpsA)
(kpsB, featuresB) = extractor.compute(grayB, kpsB)

# match the keypoints using the Euclidean distance and initialize
# the list of actual matches
rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
matches = []

# loop over the raw matches
for m in rawMatches:
  # ensure the distance passes David Lowe's ratio test
  if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
    matches.append((m[0].trainIdx, m[0].queryIdx))

# show some diagnostic information
print("# of keypoints from first image: {}".format(len(kpsA)))
print("# of keypoints from second image: {}".format(len(kpsB)))
print("# of matched keypoints: {}".format(len(matches)))


# initialize the output visualization image
(hA, wA) = imageA.shape[:2]
(hB, wB) = imageB.shape[:2]
vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
vis[0:hA, 0:wA] = imageA
vis[0:hB, wA:] = imageB

# loop over the matches
for (trainIdx, queryIdx) in matches:
  # generate a random color and draw the match
  color = np.random.randint(0, high=255, size=(3,))
  ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
  ptB = (int(kpsB[trainIdx].pt[0] + wA), int(kpsB[trainIdx].pt[1]))
  cv2.line(vis, ptA, ptB, color, 2)

  # check to see if each match should be visualized individually
  if args["visualize_each"] > 0:
    cv2.imshow("Matched", vis)
    cv2.waitKey(0)

# show the visualization
if args["visualize_each"] <= 0:
  cv2.imshow("Matched", vis)
  cv2.waitKey(0)

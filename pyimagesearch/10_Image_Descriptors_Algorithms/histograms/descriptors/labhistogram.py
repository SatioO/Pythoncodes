""" color histograms

- Learn how histograms can be used as Image descriptors.
- Apply K-means clustering to cluster color histogram features.

"""
import cv2
class LabHistogram:
    def __int__(self, bins):
        self.bins = bins

    def describe(self, image, mask = None):
        # conver the image to the L*a*b* color space, compute the histogram, and normalize it
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([lab], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist,hist).flatten()

        return hist
        

from skimage import feature
import numpy as np

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(gray, numPoints, radius, method = "uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints+3), range(0, numPoints+2))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist

    

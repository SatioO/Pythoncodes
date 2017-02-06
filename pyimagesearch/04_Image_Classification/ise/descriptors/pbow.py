# Pyramid of Bag of Visual words (PBOW)

from scipy import sparse
import numpy as np

class PBOW:
    def __init__(self, bovw, numLevels=2):
        self.bovw = bovw
        self.numLevels = numLevels

    def describe(self, imageWidth, imageHeight, kps, features):
        kpMask = np.zeros((imageHeight, imageWidth), dtype="int")
        concatHist = None

        for (i, (x, y)) in enumerate(kps):
            kpMask[y, x]=i+1

        # loop over the number of levels
        for level in np.arange(self, numLevels, -1, -1):
            numParts = 2**level
            weight = 1.0/(2**(self.numLevels-level+1))

            if level == 0:
                weight = 1.0/(2** self.numLevels)

            X = np.linspace(imageWidth/numParts, imageWidth, numParts)
            Y = np.linspace(imageHeight/numParts, imageHeight, numParts)
            xParts = np.hstack([[0], X]).astype("int")
            yParts = np.hstack([[0], Y]).astype("int")

            for i in np.arange(1, len(xParts)):
                for j in np.arange(1, len(yParts)):
                    (startX, endX) = (xParts[i - 1],xParts[i])
                    (startY, endY) = (yParts[j - 1],yParts[j])
                    idxs = np.unique(kpMask[startY:endY, startX:endX])[1:] - 1
                    hist = sparse.csr_matrix((1, self.bovw.codebook.shape[0]), dtype="float")

                    #ensure at least some features exist inside the subregion
                    if len(features[idxs]) >0:
                        hist = self.bovw.describe(features[idxs])
                        hist = weight * (hist / hist.sum())

                    if concatHist is None:
                        concatHist = hist

                    else:
                        concatHist = sparse.hstack([concatHist, hist])
        return concatHist

    def featureDim(numClusters, numLevels):
        return int(round(numClusters * (1/3.0)* ((4*(numLevels+1))-1)))

        

"""
- Process of Vector quantization
- Python script to perform vector quantization
- Python class to easily and efficiently quantize the feature vectors associated with an image into a single histogram
"""

""" An efficient BOVW implementation """
from sklearn.metrics import pairwise
from scipy.sparse import csr_matrix
import numpy as np

class BagOfVisualWords:
    def __init__(self, codebook, sparse = True):
        self.codebook = codebook
        self.sparse = sparse

    def describe(self, features):
        D = pairwise.euclidean_distances(features, Y=self.codebook)
        (words, counts) = np.unique(np.argmin(D, axis=1), return_counts=True)

        #check to see if a spare histogram should be constructed
        if self.sparse:
            hist = csr_matrix((counts, (np.zeros((len(words),)), words)),
            shape = (1, len(self.codebook)), dtype = "float")

        else:
            hist = np.zeros((len(self.codebook),),dtype="float")
            hist[words]= counts

        return hist

"""
bovw = BagOfVisualWords(vocab, sparse=False)
hist = bovw.describe(features)
"""

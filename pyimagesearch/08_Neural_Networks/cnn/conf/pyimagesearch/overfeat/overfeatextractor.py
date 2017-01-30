from sklearn_theano.feature_extraction.overfeat import SMALL_NETWORK_FILTER_SHAPES
from sklearn_theano.feature_extraction import OverfeatTransformer

class OverfeatExtractor:
    def __init__(self, layerNum):
        self.layerNum = layerNum
        self.of = OverfeatTransformer(output_layers=[layerNum])

    def describe(self, data):
        return self.of.transform(data)

    def getFeatureDim(self):
        return SMALL_NETWORK_FILTER_SHAPES[self.layerNum][0]

        

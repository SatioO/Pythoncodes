#import the necessary packages
import numpy as np

class DetectAndDescribe:
    def __init__(self, detector, descriptor):
        self.detector= detector
        self.descriptor = descriptor

    def detect(self, image, useKpList = True):
        # detect the keypoints in the image and extract local invariant descriptors
        kps = self.detector.detect(image)
        (kps, descs) = self.descriptor.compute(image, kps)

        if len(kps) == 0:
            return (None, None)

        #check to see if the keypoint should be converted to a NumPy Array
        if useKpList:
            kps = np.int0([kp.pt for kp in kps])

        # return a tuple of the keypoints and descriptor
        return (kps, descs)

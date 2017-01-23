import numpy as np
import cv2

class HSVDescriptor:
    def __init__(self, bins):
        self.bins=bins

    def describe(self, image):
        image = cv2.cvtColor(image , cv2.COLOR_BGR2HSV)
        features = []

        (h, w) = image.shape[:2]
        (cX, cY) = (int(w*0.5), int(h*0.5))

        # divide the image into 4 rectangles/segments (top-left, top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY),(cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

        # construct an elliptical mask representing the center of the image
        (axesX, axesY) = (int(w*0.75)/2, int(h*0.75)/2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (int(cX), int(cY)), (int(axesX), int(axesY)), 0, 0, 360, 255, -1)

        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            cornerMask =  np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        return np.array(features)


    def histogram(self, image, mask = None):
        # Extract a 3D color histogram from the masked region of the image, using the supplied number of bins per channel; then normalize
        # the histogram
        hist = cv2.calcHist([image], [0,1,2], mask, self.bins, [0, 180, 0, 256, 0 , 256])
        hist = cv2.normalize(hist,hist).flatten()
        return hist

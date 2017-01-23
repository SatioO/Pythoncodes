# import the necessary packages
from pyimagesearch.object_detection.helpers import *


class ObjectDetector:
    """
    - Select the model and HOG descriptor
    - For each image
       - build a pyramid
           - pyramid will give each image with decreasing size till minSize =(30,30)
           - take the scale of the image
           - now apply sliding window
               - for each window, calculate hog descriptor and apply model
               - check for minProb and it it satisfy calculate the window location on the main image and save those results

    """
    def __init__(self, model, desc):
        self.model = model
        self.desc = desc

    def detect(self, image, winDim,  winStep=4,pyramidScale=1.5, minProb = 0.7):
        boxes = []
        probs = []
        for layer in pyramid(image, scale=pyramidScale, minSize=winDim):
            scale = image.shape[0]/float(layer.shape[0])
            #loop  over the sliding windows for the current pyramid layer
            for (x, y, window) in sliding_window(layer, winStep, winDim):
                (winH, winW) = window.shape[:2]
                if winH == winDim[1] and winW == winDim[0]:
                    features = self.desc.describe(window).reshape(1, -1)
                    prob = self.model.predict_proba(features)[0][1]
                    if prob > minProb:
                        (startX, startY) = (int(scale*x), int(scale*y))
                        endX = int(startX + (scale * winW))
                        endY = int(startY + (scale * winH))
                        # update the list of bounding boxes and probabilities
                        boxes.append((startX, startY, endX, endY))
                        probs.append(prob)
        return (boxes, probs)

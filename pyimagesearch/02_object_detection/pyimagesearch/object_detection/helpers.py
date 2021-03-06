""" Custom Object detection """

"""
- Component#1 - An image Pyramid - A Multiscale representation of Image
- Component#2 - A sliding window - Scanner
"""

import imutils
import cv2

def crop_ct101_bb(image, bb, padding = 10, dstSize=(32, 32)):
    (y, h, x, w) = bb
    (x, y) = (max(x-padding,0), max(y-padding,0))
    roi = image[y:h+padding, x:w+padding]
    roi = cv2.resize(roi, dstSize, interpolation = cv2.INTER_AREA)
    return roi



def pyramid(image, scale =1.5, minSize = (30,30)):
    #yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimenstions of the image and resize it
        w = int(image.shape[1]/ scale)
        image = imutils.resize(image, width = w)

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image


def sliding_window(image, stepSize, windowSize):
    #sliding the window accross the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x,y,  image[y:y+windowSize[1],x:x+windowSize[0]])

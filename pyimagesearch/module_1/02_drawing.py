""" Drawing """


import numpy as np
import cv2

import matplotlib.pyplot as plt

canvas = np.zeros((300,300,3), dtype = "uint8")

green=(0,255,0)

"""Green Line"""
cv2.line(canvas,(0,0),(300,300),green)


""" Rectangle """
red =(255,0,0)
cv2.rectangle(canvas,(50,200),(200,225),red,5)


""" Circle """
(X,Y) = (canvas.shape[1]/2,canvas.shape[0]/2)
white=(255,255,255)

for r in xrange(0,175,25):
    cv2.circle(canvas,(X,Y),r,white)

plt.imshow(canvas)
plt.show()

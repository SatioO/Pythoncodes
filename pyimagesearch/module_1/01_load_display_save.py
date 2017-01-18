import argparse
import cv2
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and show some basic information on it
image=cv2.imread(args["image"])
print ("width: %d pixels" % (image.shape[1]))
print ("height: %d pixels" % (image.shape[0]))
print ("channels: %d pixels" % (image.shape[2]))

plt.imshow(image)
plt.show()

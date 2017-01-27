""" Histograms

- grayscale hist for thresholding
- histogram for white balancing
- histograms for object tracking in images (Camshift Algo)
- HOG and SIFT descriptors
- bag of visual words
- image search engines and machine learning


cv2.calcHist(images , channels, mask , histSize, ranges)
"""

hist = cv2.calcHist([image], [0],None,[256], [0,256])

plt.figure()
plt.title("Grayscale histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.plot(hist)
plt.xlim([0,256])

#normalization of histograms
hist /= hist.sum()

#COLOR Histograms
chans = cv2.split(image)
colors = ("b", "g", "r")

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0,256])
    plt.plot(hist, color = color)
    plt.xlim([0, 256])


# let's move on to 2D histograms — we need to reduce the
# number of bins in the histogram from 256 to 32 so we can
# better visualize the results
fig = plt.figure()

# plot a 2D color histogram for green and blue
ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32],
  [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for G and B")
plt.colorbar(p)

# plot a 2D color histogram for green and red
ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32],
  [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for G and R")
plt.colorbar(p)

# plot a 2D color histogram for blue and red
ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32],
  [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for B and R")
plt.colorbar(p)

# finally, let's examine the dimensionality of one of the 2D
# histograms
print "2D histogram shape: %s, with %d values" % (
  hist.shape, hist.flatten().shape[0])



""" Histogram Equalization
 - improves the contrast of an image by "stretching" the distribution of pixels.
 """
eq = cv2.equalizeHist(image)


## Histograms for only masks
def plot_histogram(image,title,mask=None):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of pixels")

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0,256])
        plt.plot(hist, color = color)
        plt.xlim([0, 256])
    

###Quantifying and abstractly representing an image using only a list of numbers  The process of quantifying an Image is called feature extraction.

Image descriptors and feature descriptors govern how an image is abstracted and quantified, while feature vectors are the output of descriptors and used to quantify the image. Taken as whole, this process is called feature extraction

**Uses**:
..1. to compare to image for similarity
..2. to rank images in search results when building an image search engine
..3. to use when training an image classifier to recognize the contents of an image

**Terms**:
..1. Image descriptors
..2. Feature descriptors
..3. Feature Vectors

**Input Image ---> (Feature/Image descriptor) ---> Feature Vectors**

**Image Feature Vector**:
... An abstraction of an Image used to characterize and numerically quantify the contents of an Image, Normally real, integer or binary valued. Simply put, a feature vector is a list of numbers used to represent an image.

**Descriptors**:
... Algorithms and methodologies used to extract feature vectors are called image descriptors and feature descriptors.

... **Image Descriptor**: An image descriptor is an algorithm and methodology that governs how an input image is globally quantified and ... returns a feature vector abstractly representing the image content
... Example: color channel statistics, color histograms and Local Binary Patterns
... Problems : Its not robust to changes in how the image is rotated , translated, or how viewpoints of an image change

... **Feature Descriptor**: A feature descriptor is an algorithm and methodology that governs how an input region of an image is locally
... quantified. A feature descriptor accepts a single input image and governs multiple feature vectors.  
... Example : keypoint detectors, SIFT, SURF, ORB, BRISK, BRIEF, and FREAK

* Image descriptor: 1 image in , 1 feature vector out
* Feature descriptor: 1 image in , many feature vector out

#Image Descriptors:
**Color Channel statistics**
**Color Histograms**
**Hu Moments**
**Zernike Moments**
**Haralick texture**
**Local Binary Patterns**
**Histogram of Oriented Gradients**

#Feature Descriptors:
##keypoint detectors
**FAST**
**Harris**
**GFTT**
**DoG**
**Fast Hessian**
**STAR**
**MSER**
**Dense**
**BRISK**
**ORB**


##Local Invariant descriptor
**SIFT**
**RootSIFT**
**SURF**
**Real-valued feature extraction and matching**

##BINARY Descriptors
**What are Binary descriptors?**
**BRIEF**
**ORB(descriptor)**
**BRISK(descriptor)**
**FREAK**
**Binary feature extraction and matching** 

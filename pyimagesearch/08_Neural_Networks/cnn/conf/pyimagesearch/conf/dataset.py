import numpy as np
import cv2

def prepare_image(image, fixedSize):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, tuple(fixedSize))

    return image


def build_batch(paths, fixedSize):
    images = [prepare_image(cv2.imread(p), fixedSize) for p in paths]
    images = np.array(images, dtype="float")

    labels = [":".join(p.split("/")[-2:]) for p in paths]

    return (labels, images)

def chunk(l, n):
    for i in np.arange(0, len(l), n):
        yield l[i:i+n]

""" building a mini fashion search engine using texture

| --- local_binary_pattern
|     | --- __init__.py
|     | --- descriptors
|     |    | ----__init__.py
|     |    | ---localbinarypattern.py
| --- search_shirts.py

"""


""" LBPs
- Successfully used in Face recognition
- Implemented in both mahotas and scikit-image

using scikit-image:
from skimage import feature
numPoints = 24
radius = 3

lbp = feature.local_binary_pattern(gray, numPoints, radius, method = "uniform")
(hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints+3), range(0, numPoints+2))

eps = 1e-7
hist = hist.astype("float")
hist /= (hist.sum() + eps)

import mahotas
hist = mahotas.features.lbp(gray, radius, points)

"""

""" Translation

  [[1, 0 ,x],
  [0, 1, y]]

"""
def translate(image,x,y):
    M = np.float32([[1,0,x],[0,1,y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

""" Rotation

[[cosθ , -sinθ],
[sinθ, cosθ ]]

"""
def rotate(image, angle, center=None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w,h))

    return rotated

""" Resizing

interpolation:
  - cv2.INTER_NEAREST
  - cv2.INTER_LINEAR (default by cv2) - upsampling
  - cv2.INTER_AREA (default in our function) - downsampling

"""

def resize(image, width= None, height = None, inter= cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height/float(h)
        dim = (int(w,r), height)

    else:
        r = width /float(w)
        dim = (width,int(h*r))

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized

""" Flipping

h = 0 - horizontal
h = 1 - vertical
h = -1 - both

"""


""" Cropping

Use numpy array slicing

"""

""" Image arthemetic
- 8 Bit unsigned integers [0, 255]

- Numpy : perform modulus arithmetic and "wrap around" - np.uint8
- Opencv : ensure pixel values never fall outside the range [0,255] - cv2.add(np.uint8([50]))

Add 100 and we can increase the intensity of the pixels
subtract 50 and you can dark the image

"""


""" Bitwise operations

- AND
- OR
- XOR
- NOT

"""

"""  Masking """

""" Splitting and merging channels

B, G, R = cv2.split([B, G, R])
image = cv2.merge([B, G, R])

"""



# auto_canny
def auto_canny(image, sigma = 0.33):
    v = np.median(image)
    lower = int(max(0,(1.0-sigma)*v))
    upper = int(max(0,(1.0+sigma)*v))
    edged = cv2.Canny(image, lower, upper)
    return edged

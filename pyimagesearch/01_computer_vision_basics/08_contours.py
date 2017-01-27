##
""" # Contours
- Centroid/Center of Mass
- Area
- Perimeter
- Bounding boxes
- Rotated Bounding boxes
- Minimum enclosing circles
- Fitting an ellipse
"""

import cv2
import matplotlib.pyplot as plt


image = cv2.imread("shapes.png")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

(cnts,_) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clone = image.copy()

for c in cnts:
    M = cv2.moments(c)
    print M
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])

    cv2.circle(clone, (cX,cY), 10, (0,255,0),-1)


plt.imshow(clone)
plt.show()

# Area and Perimeter
for (i,c) in enumerate(cnts):
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c,True)
    print ("Contour #%d -- area: %.2f, perimeter: %.2f" % (i +1, area, perimeter))
    cv2.drawContours(clone, [c], -1, (0,255,0), 2)

    M = cv2.moments(c)
    cX = int(M["10"]/M["m00"])
    cY = int(M["01"]/M["m00"])
    cv2.putText(clone, "#%d" % (i +1), (cX - 20, cY),cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 4)

    cv2.imshow("Contours",clone)
    cv2.waitKey(0)


# Bounding box
clone = image.copy()

for c in cnts:
    (x, y, w, h) = cv2.boundingRect()
    cv2.rectangle(clone, (x,y), (x+w, y+h), (0, 255, 0), 2)


# Rotated Bouding boxes
for c in cnts:
    box = cv2.minAreaRect()
    box = np.int0(cv2.cv.BoxPoints(box))
    cv2.drawContours(clone, [box], -1, (0, 255,0), 2)

#minimum enclosing circles
for c in cnts:
    ((x,y), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(clone, (int(x), int(y)), int(radius), (0,255,0), 2)


# fitting an ellipse
for c in cnts:
    if len(c) >= 5:
        ellipse = cv2.fitEllipse(c)
        cv2.ellipse(clone , ellipse, (0,255,0),2)


""" # Advanced Contour Properties

- Aspect Ratio - image width / image height
- Extent - shape area/ bounding box area
- Convex hull
- Solidity - contour area/ convex hull area

"""
(cnts,_) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for (i,c) in enumerate(cnts):
    area = cv2.contourArea(c)
    (x, y , w, h) = cv2.boundingRect(c)

    hull = cv2.convexHull(c)
    hullArea = cv2.contourArea(hull)
    solidity = area/float(hullArea)

    # initialize the character vector
    char = "?"
    if solidity > 0.9:
        char="o"
    elif solidity > 0.5:
        char = "X"

    if char != "?":
        cv2.drawContours(image,[c],-1,(0,255,0),3)
        cv2.putText(image, char, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.25,(0, 255,0), 4)

# - HU Moments
# - Zernike Moments
""" CONTOUR APPROXIMATION

- Ramer - Douglas - Peucker
- Split and merge Algorithm

"""
for c in cnts:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.005*peri,True)
    if len(approx) == 4:
        cv2.drawContours(image,[c],-1,(0,255,255), 2)
        (x,y,w,h) = cv2.boundingRect(approx)
        cv2.putText(image,"Rectangle", (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,255,255), 2)


def sort_contours(cnts, method = "left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse= True
    if method == "left-to-right" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse = reverse))

    return (cnts,boundingBoxes)

def draw_contour(image, c, i):
    M = cv2.moments(c)
    cX = int(M["10"]/M["m00"])
    cY = int(M["01"]/M["m00"])
    cv2.putText(image, "#%d" % (i +1), (cX - 20, cY),cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 4)
    return image


##### - Detect contours and sort them
image = cv2.imread(args["image"])
accumEdged = np.zeros(image.shape[:2],dtype="uint8")

for chan in cv2.split(image):
    chan = cv2.medianBlur(chan, 11)
    edged = cv2.Canny(chan, 50 , 200)
    accumEdged = cv2.bitwise_or(accumEdged, edged)
    

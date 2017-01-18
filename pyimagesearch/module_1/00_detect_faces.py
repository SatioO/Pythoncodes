# Detect a face using opencv

import cv2

def detect_face(image_loc):
    image = cv2.imread(image_loc)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # load the face detector and detect faces in the image
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    rects = detector.detectMultiScale(gray,scaleFactor=1.05, minNeighbors=7,
    minSize=(30,30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

    for (x, y, w ,h in rects):
        cv2.rectangle(image,,(x,y),(x+w,y+h),(0,255,0), 2)

    return image

#cv2.imshow("Faces",image)
#cv2.waitkey(0)

import cv2

class DistanceFinder:
    def __init__(self, knownWidth, knownDistance):
        self.knownWidth = knownWidth
        self.knownDistance = knownDistance

        #initialize the focal length
        self.focalLength = 0

    def calibrate(self, width):
        self.focalLength = (width * self.knownDistance) / self.knownWidth

    def distance(self, perceivedWidth):
        return (self.knownWidth * self.focalLength) / perceivedWidth


    def findSquareMarker(image):
        #convert the image to grayscale blur it and find edges in the images
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GuassianBlur(gray, (5,5),0)
        edged = cv2.Canny(gray, 35, 125)


        # find contours in the edged images, sort them according to their area (from largest to smallest) and initialize the marker dimension
        (cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key  = cv2.contourArea, reverse = True)
        markerDim = None

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)

            #ensure the contour is a rectangle
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                aspectRatio = w/float(h)

                #check to see if the the aspectRatio is within tolerable limits
                if aspectRatio >0.9 and aspectRatio < 1.1:
                    markerDim = (x, y, w, h)
                    break

        return markerDim


    def draw(image, boundingBox, dist, color=(0,255,0), thickness = 2):
        (x, y, w, h) = boundingBox
        cv2.rectangle(image, (x,y), (x+w, y+h), color, thickness)
        cv2.rectangle(image, (x+w, y+h), color, thickness)
        cv2.putText(image, "%.2fft"% (dist/12),(image.shape[1]-200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)

        return image

        

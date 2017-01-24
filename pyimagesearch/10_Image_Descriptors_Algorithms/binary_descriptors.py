## What are binary descriptors?

#BRIEF 
detector = cv2.FeatureDetector_create("FAST")
extractor = cv2.DescriptorExtractor_create("BRIEF")

image = cv2.imread(arg["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kps = detector.detect(gray)
(kps, descs) = extractor.compute(gray, kps)

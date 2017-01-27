Two Phase Process:
- Phase 1: Detect the presence of faces in an image or video stream using methods such as Haar cascades, HOG + Linear SVM , Deep Learning, or any other algorithm that can localize faces.
- Phase 2: Take each of the faces detected during localization phase and identify each of them - this is where we actually assign a name to a face.

Algorithms for Face Recognition:
- The Eigenfaces algorithm - PCA
- Fisherfaces - LDA
- Local Binary Pattern


|--- cascades
|    | ---- haarcascade_frontalface_default.xml
| --- output
|    | ---- classifier
|    | ---- faces
| --- pyimagesearch
|    | ---- __init__.py
|    | ---- face_recognition
|    |      | --- __init__.py
|    |      | --- datasets.py
|    |      | --- facedetector.py
|    |      | --- facerecognizer.py
|    | ---- resultsmontage.py
| --- gather_selfies.py
| --- recognize.py
| --- train_recognizer.py

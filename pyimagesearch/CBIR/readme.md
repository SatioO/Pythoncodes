###Objectives
 - The three types of image search engines
 - Important terms such as feature extraction, feature vector , indexing , distance metrics, querying and result set
 - The 4 steps of building any CBIR system
 - How to evaluate CBIR system
 - The difference between CBIR and machine learning

The four steps of any CBIR system
1.  Image Descriptor
     - color of the Image
     - shape of an object in the Image
         - Hu Moments
         - Zernike Moments
     - characterize texture
         - Haralick texture descriptor
         - Local Binary Patterns
         - HOG
         - Fourier an Wavelet transformation of a grayscale image

2. Feature Extraction and indexing your dataset
3. Define your similarity metric (conditions : 1) Non-negative, 2) Coincidence Axiom 3) Symmetry 4) Triangle Inequality )
      - Euclidean distance
      - Cosine distance
      - chi-square distance
      - Manhattan/City block
      - Intersection : sum of minimum entries between histograms h1 and h2
      - Hamming : It measures the number of disagreements between the two vectors, then divides by the length  
4. Indexing and specialized data structures
     - inverted indexes
     - kd-trees
     - random projection forests


Dataset of Images ------> Extract Features from each Image -------> Store Features


###Performing a search
User Submits Query Image ------> Extract Features from Query Image -----> Compare Query Features to Image Features in Database <----- DataBase of Features
                                                                                      |
                                                                                      |
                                                                                      |
                                                                                      \/
                              Display Results of the user              <----sort Results by relevancy



###Evaluating a CBIR system
- Precision = Num of relevant images retrieved / total Num of images retrieved from the database
- Recall = Num of relevant images retrieved / total num of relevant images in Database
- f-score = 2*(Precision * recall / Precision+recall)


| --- pyimagesearch
       | -- _init_.py
       | -- CBIR
       |    | --- __init__.py
       |    | --- dists.py
       |    | --- hsvdescriptor.py
       |    | --- resultsmontage.py
       |    | --- searcher.py
| --- index.py
| --- search.py



Buidling a bag of visual words
* step 1: Feature Extraction
  + Input Image ----> Feature Descriptor ---> vector
  + Detecting keypoints and extracting SIFT features from salient regions of the images.
- step 2: Codebook construction.
  + K-means clustering algorithm.
- step 3: Vector quantization.
  + Extract feature vectors from the image in the same manner as step #1 above.
  + For each extracted feature vector, compute its nearest neighbour in the dictionary created in step #2. this is normally accomplised using Eculidean distance.
  + Take the set of nearest neighbour labels and build a histogram of length k (the number of clusters generated from k-means).

""" EXTRACTING KEYPOINTS AND LOCAL INVARIANT DESCRIPTORS

- Extracting keypoints and features from the image dataset
- clustering the extracted features usng k-means to form a codebook
- constructing bag of visual words (BOVW) representation for each image by quantizing the feature vectors associated with each image into
a histogram using the codebook in step2.
- Accepting a query image from the user, constructing the BOVW representation for the query, and performing the actual search.

"""


"""

| --- ise
       | -- _init_.py
       | -- descriptors
       |    | --- __init__.py
       |    | --- detectanddescribe.py
       |    | --- pbow.py
       |    | --- rootsift.py
       | -- indexer
       |    | --- __init__.py
       |    | --- bovwindexer.py
       |    | --- baseindexer.py
       |    | --- featureindexer.py
       | -- ir
       |    | --- __init__.py
       |    | --- bagofvisualwords.py
       |    | --- vocabulary.py
| --- cluster_features.py
| --- index_features.py
| --- extract_pbow.py
| --- sample_datasets.py
| --- test_model.py
| --- train_model.py 

"""

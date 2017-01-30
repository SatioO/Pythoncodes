""" The OverFeat Framework """

"""
Layer                    1           2           3       4                  5          6        7        8
stage                conv+max    conv+max      conv     conv             conv+max     full     full     full
channels                96         256          512     1024               1024       3072     4096     1000
filter size            11*11       5*5          3*3      3*3               3*3         -        -         -
conv stride            4*4         1*1          1*1      1*1               1*1         -        -         -
pooling size           2*2         2*2           -        -                2*2         -        -         -
pooling stride         2*2         2*2           -        -                2*2         -        -         -
zero padding size       -           -          1*1*1*1  1*1*1*1          1*1*1*1       -        -         -
spatial input size     231*231     24*24        12*12    12*12            12*12       6*6      1*1       1*1

RELU activation
SGD momentum 0.6, weight decay = 1*10^-5
learning rate 5*10^-2
"""

"""
1.Extract CNN features from images.
2.serialize the features to disk in an efficient HDF5 dataset.
3.Train a classifier on these feature representations.
"""

"""
| --- conf
| --- pyimagesearch
       | -- __init__.py
       | -- overfeat
       |    | --- __init__.py
       |    | --- overfeatextractor.py
       | -- indexer
       |    | --- __init__.py
       |    | --- bovwindexer.py
       |    | --- baseindexer.py
       |    | --- featureindexer.py
       | -- utils
       |    | --- __init__.py
       |    | --- conf.py
       |    | --- dataset.py
| --- index_features.py
| --- test.py
| --- train.py
"""

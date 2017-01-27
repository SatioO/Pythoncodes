from ise.ir.bagofvisualwords import *
from ise.indexer.bovwindexer import *
import argparse
import cpickle #check for python 3.5
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True, help = "Path to features database")
ap.add_argument("-c", "--codebook", required=True, help = "Path to the codebook")
ap.add_argument("-b", "--bovw-db", required=True, help = "Path to where the bag of visual words are stored")
ap.add_argument("-s", "--max-buffer-size", type=int, default=500, help = "Maximum buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())


#load the codebook vocabulary and initialize the bag-of-visual-words transformer
vocab = cPickle.loads(open(args["codebook"]).read())
bovw = BagOfVisualWords(vocab)

# open the features database and initialize the bag-of-visual-words indexer
featuresDB = h5py.File(args["features_db"], mode="r")
bi = BOVWIndexer(bovw.codebook.shape[0], args["bovw_db"], estNumImages=featuresDB["image_ids"].shape[0],
maxBufferSize = args["max_buffer_size"])


for (i, (imageID, offset)) in enumerate(zip(featuresDB["image_ids"], featuresDB["index"])):
    if i>0 and i%10 ==0:
        bi._debug("processed {} images".format(i), msgType="[PROGRESS]")

        #extract the feature vectors for the current image using the starting and ending offsets (while ignoring the keypoints) and
        #then quantize the bag of visual words histogram
        features = featuresDB["features"][offset[0]:offset[1]][:,2:]
        hist = bovw.describe(features)

        #add the bovw to the index
        hist /= hist.sum()
        bi.add(hist)

#close the features db and finish the indexing process
featuresDB.close()
bi.finish()

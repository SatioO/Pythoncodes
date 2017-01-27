from ise.ir.vocabulary import Vocabulary
import argparse
import cPickle #check for python3

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True, help = "Path to features database")
ap.add_argument("-c", "--codebook", required=True, help = "Path to the codebook")
ap.add_argument("-k", "--clusters", type = int, default = 64, help = "# of clusters to generate")
ap.add_argument("-p", "--percentage", type = float, default = 0.25, help = "percentage of total features to use when clustering")
args = vars(ap.parse_args())


#create the visual words vocabulary
voc = Vocabulary(args["features_db"])
vocab = voc.fit(args["clusters"], args["percentage"])

#dump the clusters to file
f = open(args["codebook"], "w")
f.write(cPickle.dumps(vocab))
f.close()

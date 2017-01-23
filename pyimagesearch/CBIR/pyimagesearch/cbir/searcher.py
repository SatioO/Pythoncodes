#
import dists
import csv

class Searcher:
    def __init__(self, dbPath):
        self.dbPath= dbPath  #store the database path

    def search(self, queryFeatures,  numResults=10):
        #initialize the results dictionary
        results = {}
        with open(self.dbpath) as f:
            reader = csv.reader(f)

            for row in reader:
                features = [float(x) for x in row[1:]]
                d = dists.chi2_distance(features, queryFeatures)
                results[row[0]] = d
            #close the reader
            f.close()
        results = sorted([(v, k) for (k, v) in results.items()])

        #return the results
        return results[:numResults]
        

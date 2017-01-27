from ise.indexer.baseindexer import BaseIndexer
from scipy import sparse
import numpy as np
import h5py

class BOVWIndexer(BaseIndexer):
    def __init__(self, fvectorSize, dbPath, estNumImages=500, maxBufferSize=500, verbose=True):
        super(BOVWIndexer, self).__init__(dbPath, estNumImages=estNumImages, maxBufferSize=maxBufferSize, dbResizeFactor=dbResizeFactor,
        verbose=verbose)

        self.db = h5py.File(self.dbPath, mode="w")
        self.bovwDB = None
        self.bovwBuffer = None
        self.idxs = {"bovw":0}

        self.fvectorSize = fvectorSize
        self._df = np.zeros((fvectorSize,), dtype="float")
        self.totalImages = 0

    def add(self, hist):
        self.bovwBuffer = BaseIndexer.featuresStack(hist, self.bovwBuffer, stackMethod = sparse.vstack)
        self._df[np.where(hist.toarray()[0] > 0)] +=1

        #check to see if we have reached the Maximum Buffer size
        if self.bovwBuffer.shape[0] >= self.maxBufferSize:
            if self.bovwDB is None:
                self._debug("inital buffer full")
                self._createDatasets()

            self.writeBuffers()

    def _writeBuffers(self):
        if self.bovwBuffers is not None and self.bovwBuffers.shape[0] >0:
            self._writeBuffer(self.bovwDB,"bovw", self.bovwBuffer, "bovw", sparse=True)
            self.idxs["bovw"] += self.bovwBuffer.shape[0]
            self.bovwBuffer = None

    def _createDatasets(self):
        self._debug("create datasets...")
        self.bovwDB = self.db.create_dataset("bovw", (self.estNumImages,self.fvectorSize), maxshape=(None,self.fvectorSize), dtype = "float")

    def finish(self):

    # if the databases have not been initialized, then the original# buffers were never filled up
        if self.bovwDB is None:
            self._debug("minimum init buffer not reached", msgType="[WARN]")
            self._createDatasets()
        # write any unempty buffers to file
        self._debug("writing un-empty buffers...")
        self._writeBuffers()

        # compact datasets
        self._debug("compacting datasets...")
        self._resizeDataset(self.bovwDB, "bovw", finished=self.idxs["bovw"])

        # close the database
        self.totalImages = self.bovwDB.shape[0]
        self.db.close()

    def db(self, method = None):
        if method == "idf":
            return np.log(self.totalImages/(1.0+self._df))
        return self._df

        

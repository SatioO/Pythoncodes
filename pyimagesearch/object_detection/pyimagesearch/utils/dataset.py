import numpy as np
import h5py


def dump_dataset(data, labels, path, datasetName, writeMethod="w"):
    db = h5py.File(path, writeMethod)
    dataset = db.create_dataset(datasetName, (len(data), len(data[0])+1), dtype="float")
    dataset[0:len(data)]=np.c_[labels, data]
    db.close()

def load_dataset(path, datasetName):
    db = h5py.File(path, "r")
    (labels, data)=(db[datasetName][:,0], db[datasetName][:,1:])
    db.close()

    return (data,labels)

- Caffe is not a true "library" in a Pythonic sense of the word.
- Its a suite of tools we can use to train Deep learning networks, with a heavy focus placed on CNNs
- Caffe is extremely fast, capable of processing of over 60M images per day o NVIDIA K40
- Caffe doesn't usually operate on raw image files. key-value storage database such as Lightning Memory-Mapped Database(LMDB) or LevelDB.

| --- pyimagesearch
|    | --- __init__.py
|    | --- utils
|    |    | --- __init__.py
|    |    | --- dataset.py
| --- test_images
| --- cifar10_deploy.prototxt - (saves the architecture of our model)
| --- cifar10_solver.prototxt - (contains configurations on how to train the network)
| --- cifar10_train_test.prototxt - (includes training and testing specific configurations)
| --- convert_cifar10.py - (deserialize it into a collection of images)
| --- create_dataset.sh - (take our set of images on disk and build the database required by Caffe)
| --- make_dataset_mean.sh - (take mean of the entire dataset)
| --- test.py - (evaluate the network)
| --- train.sh - (train the network)

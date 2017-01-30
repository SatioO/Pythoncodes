"""
AutoEncoders:
An Autoencoder is a feedforward NN that attempts to learn a compressed representation of a dataset. This network architecture consists of one input layer, at least one hidden layer, and finally an output layer:

Input (5) - Hidden(4) - output(5)

In essence, an Autoencoder is trained to "reconstruct" its input - we're essentially trying to obtain (almost) the same output from the network as we put into it, but "compressed" in some manner.

Our goal is not for Autoencoder architecture to learn a direct mapping from input to output , but rather learn the structure of the data itself. In order for this to happen, the number of nodes in the hidden layer should be smaller than the size of the input and output layers, forcing the network to learn only the utmost important and discriminative features. The most discriminative features learned by the network also serve as a form of dimensionality reduction.

Restricted Boltzmann Machines (RBM):
RBMs are a generative, stochastic network that can learn a probability distribution over a set of inputs. Unlike traditional feedforward networks, the connection between visible and hidden layers of the RBM are undirected, implying that "information" can travel in both the visible-to-hidden and hidden-to-visible directions

Use Contrastive divergence Algorithm
Phases:
 - A Positive Phase : An input sample v is presented to the input layer. Denote hidden layer activation as h
 - A Negative Phase : We take h and propagate it back through the visible layer, called v'
 - A weight update Phase :
 w(t+1) = w(t) + \alpha *(v*h(T) - v'*h'(T))

 The Positive phase of Contrastive divergence (v and h) reflect the network's initial representation of the original input vectors.
 The Negative phase on the other hand attempts to reconstruct the original input vector (v' and h'). The goal is for the generated data to be as close as possible to the original input data.

 """

 from sklearn.neural_network import BernoulliRBM
 import matplotlib.pyplot as plt
 from sklearn import datasets

 digits = datasets.load_digits()
 data = digits.data.astype("float")
 data = (data - data.min(axis=0)) / (data.max(axis = 0) + 1e-5)

 rbm = BernoulliRBM(n_components=64, learning_rate=0.05, n_iter=20, random_state = 42, verbose = True)
 rbm.fit(data)

 #initialize the plot
 plt.figure()
 plt.suptitle("64 MNIST components extracted by RBM")

 for (i, comp) in enumerate(rbm.components_):
     plt.subplot(8, 8, i+1)
     plt.imshow(comp.reshape((8,8)), cmap = plt.cm.gray_r, interpolation = "nearest")
     plt.xticks([])
     plt.yticks([])

plt.show()

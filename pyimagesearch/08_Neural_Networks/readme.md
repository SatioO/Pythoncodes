#Neural Network

##Biological Process of neuron:  
- Brain consists of approx 10 billion neurons , each connected to about 10,000 neurons.
- The cell body of the neuron is called the soma, where the inputs (dendrites) and outputs (axons) connect soma to other soma. Each neuron receives electrochemical inputs from other neurons at their dendrites.
- if these electrical inputs are sufficiently powerful to activate the neuron, then the activated neuron transmits the signal along its axon, passing it along to the dendrites of other neurons. These attached neurons may also fire, and thus continue the process of firing
and passing the message along.


##Neural Learning  
When an axon of cell A is near enough to excite cell B, and repeatedly or persistently takes place in firing it, some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased - **The Organization of Behavior**.

##Neural Network used in  
**Classification**  
**Regression**  
**Clustering**  
**Vector Quantization**  
**Pattern Association**  


##Activation functions
- step function
- sigmoid
- tanh
- ReLU
- Leaky ReLU

##Loss Functions
- Mean Squared error : (y-p)^2
- Exponential Log likelihood : p * (y*log(p))
- Cross Entropy : ((y*log(p)) + (1-y)) * (1-log(p))
- Negative Log likelihood : -y * log(p)
- RMSE Cross Entropy : ((y-p)^2)^0.5
- Squared loss : 0.5 * (y - p)^2

##Softwares
- Install opencv  
- Install caffe
- Install sklearn-theano
- Install nolearn
- Install keras


architecture
| --- pyimagesearch
|     | --- __init__.py
|     | --- utils
|     |     | --- __init__.py
|     |     | --- datasets.py
| --- cifar_dbn.py
| --- mnist_dbn.py
| --- plot_dbn.py



CNNs give two benefits:
1) Local Invariance
2) Compositionality

Rule of thumb:
- The input layer should be divisible by 2 multiple times. Common choices include 32, 64, 224, 384, 512
- CONV layers should use small filter sizes of 3*3 or 5*5, along with a stride of S=1.
- Pad the input values with Zero-Padding
- Pool layers to reduce the spatial dimensions of the input . maxpooling (2*2) with stride=2

Popular frameworks
- LeNet
- AlexNet
- ZFNet
- GoogleNet
- VGGNet

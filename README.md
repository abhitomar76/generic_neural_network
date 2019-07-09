### Generic Neural Network architecture
Implementation of neural network from scratch(without any library NN API)

## Objective: 
To build a library to implement a generic neural network architecture i.e NN architecture is built at run time according to the 
“number of layers” and “number of neurons at each layer” passed as argument to the program. We will learn about the following after 
this exercise:

•	Implementation of Generic Neural Network (without using any NN python library APIs)

•	One-hot encoding concept & generation

•	Back propagation algorithm implementation to train the network using SGD

•	Confusion matrix (implementation & interpretation)

•	“sigmoid” and “relu” as the activation functions

•	A small exercise for the reader to optimize the cost while using “relu” function

[Note: refer [link1](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) and [link2](https://en.wikipedia.org/wiki/Confusion_matrix) to learn more about confusion matrix]

## Methodology Used: 
The network is trained using the Stochastic  Gradient Descent (SGD) algorithm with batch size as the configurable input to the program.
## Data set Used (Training/Testing):
Generic Neural Network is trained and tested over the [Poker Hand](https://archive.ics.uci.edu/ml/datasets/Poker+Hand) dataset available on the UCI repository. The training set contains 
25010 examples whereas the test set contains 1000000 examples each. The dataset consists of 10 categorical attributes. The last entry 
in each row denotes the class label. More details about the data set are available in detail over the link given above.
## Problem Statement:
Solve the following problems “a-e” detailed below by implementing the Neural Network architecture as per the details below:
(a) The Poker Hand dataset described above in “Data set Used” section has 10 categorical attributes. Transform and save the given train 
and test sets using one hot encoding to convert categorical features to binary. We will use these new train and test sets for the 
subsequent parts. 
(b) Write a program to implement a generic neural network architecture. Implement the backpropagation algorithm to train the network. We will train the network using Stochastic Gradient Descent (SGD) where the batch size is an input to the program. Program shall accept the following parameters: 

•	the size of the batch for SGD

•	the number of inputs 

•	a list of numbers where the size of the list denotes the number of hidden layers in the network and each number in the list denotes the number of units (perceptrons) in the corresponding hidden layer. Eg. a list [100 50] speciﬁes two hidden layers; ﬁrst one with 100 units and second one with 50 units. 

•	the number of outputs i.e the number of classes Assume a fully connected architecture i.e., each unit in a hidden layer is connected to every unit in the next layer. You should implement the algorithm from ﬁrst principles and not use any existing MATLAB/python modules. Use the sigmoid function as the activation unit. 

(c) In this part, we use the above implementation to experiment with a neural network having a single hidden layer. Vary the number of hidden layer units from the set {5, 10, 15, 20, 25}. Set the learning rate to 0.1. Choose a suitable stopping criterion and report it. Observe and plot the accuracy on the training and the test sets, time taken to train the network. Plot the metric on the Y axis against the number of hidden layer units on the X axis. Additionally, report the confusion matrix for the test set, for each of the above parameter values. What do you observe? How do the above metrics and the confusion matrix change with the number of hidden layer units? 

(d) In this part, we will experiment with a network having two hidden layers, each having the same number of neurons. Set the learning rate to 0.1 and vary the number of hidden layer units, as described in part (c). Report the metrics and the confusion matrix on the test set, as described in the previous part. How do the metrics and the confusion matrix change with the number of hidden layer units? What eﬀect does increasing the number of hidden layers, keeping the number of hidden layer units same, have on the metrics and the confusion matrix? 

(e) In this part, we will use ReLU as the activation instead of the sigmoid function, only in the hidden layer(s). ReLU is deﬁned 
 using the function: g(z) = max(0, z). Change your code to work with the ReLU activation unit. Make sure to correctly implement 
 gradient descent by making use of subgradient at z = 0. Here is a [resource](https://en.wikipedia.org/wiki/Subderivative) to know more about sub-gradients. Repeat part (e) 
 using ReLU as the activation function in the hidden layers and report the metrics and the confusion matrix and described previously. 
 What eﬀect does using ReLU have on each of the metrics as well as the confusion matrix?
 
## Implementation:
File “nn_main.py” contains the control code to test the solution for parts “a-e” as specified in the “Problem Statement”. Follow the instructions in the "Readme.docx" to run the code.

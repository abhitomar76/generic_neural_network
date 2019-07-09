# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:28:31 2019

@author: abhishekt
"""


import random
import time
import datetime
import numpy as np
from nn_utils import *

"""
Function to create the vectors for bias & weights

In - layers_dims - List of activation units for each layer
"""
def init_params(layers_dims):

    biases=[]
    weights=[]
    
    #create vectors & randomly initialize
    for i in range(1,len(layers_dims)):
        #bias for all layers starting from 1st hidden layer
        biases.append(np.random.randn(layers_dims[i], 1))
        weights.append(np.random.randn(layers_dims[i], layers_dims[i-1]))

    return biases, weights

"""
Interface function that implements a generic Neural Network architecture based
on the specifications provided in param - 'layers_dims'
Input Parameters:
    train_X - training data
    train_Y - training labels
    layers_dim - list of dimensions of activation units at each layer
                e.g [85 30 10] architecture is like In-Hidden-Out as 85*30*10
    epoch - epoch count (no. of iterations for which SGD will run)
    batch_size - batch size of samples for each SGD cycle
    non_linear_operation - could be 'sigmoid' or 'relu' for hidden units
    learn_rate - hyperparameter for learning rate that SGD uses
    adaptive_learning_rate - could be 'fixed' or 'variable'. When 'variable',
                it will reduce the learning rate by factor of 5 each time the 
                cost error fails to reduce below 'tol' of 1e-4
                
Output:
    Returns training & test accuracy
"""
def generic_NN_model(train_X, train_Y, test_X, test_Y, layers_dims, epoch=30, \
                  batch_size=10, non_linear_operation = 'sigmoid', \
                  learn_rate = 0.05, adaptive_learning_rate='fixed'):
    
    print("No. of layers", len(layers_dims))
    print("NN architecture of units", layers_dims)

    #initialize the bias & weight vectors for the NN layers    
    biases, weights = init_params(layers_dims)
    
    #Train model using the batch Gradient Descent
    #param - 'batch_size' specifies the size of batch on which GD will be performed
    train_accuracy, test_accuracy = sgd_algo(train_X, train_Y, layers_dims, epoch, biases, weights, batch_size, \
                            non_linear_operation, learn_rate, adaptive_learning_rate, \
                            test_X, test_Y)
    
    return train_accuracy, test_accuracy

"""
Implementation of Stochastic Gradient Descent algorithm
Input Parameters:
    train_X - training data
    train_Y - training labels
    layers_dim - list of dimensions of activation units at each layer
                e.g [85 30 10] architecture is like In-Hidden-Out as 85*30*10
    epoch - epoch count (no. of iterations for which SGD will run)
    biases, weights - learned parameters
    mini_batch_size - batch size of samples for each SGD cycle
    non_linear_operation - could be 'sigmoid' or 'relu' for hidden units
    learn_rate - hyperparameter for learning rate that SGD uses
    adaptive_learning_rate - could be 'fixed' or 'variable'. When 'variable',
                it will reduce the learning rate by factor of 5 each time the 
                cost error fails to reduce below 'tol' of 1e-4
    test_X - test data set
    test_Y - test data labels
Output:
    Returns training & test accuracy   

[Potential improvement - if time permits, this function would be refactored as
             follow -
             1. to return the learning parameters
             2. move the accuracy calculation part to generic_NN_model()
"""
def sgd_algo(train_X, train_Y, layers_dims, epochs,  biases, weights, mini_batch_size, non_linear_operation='sigmoid', \
                learn_rate=0.1, adaptive_learning_rate='fixed',\
                test_X=None, test_Y=None):

    #training data dimensions
    n_samples, n_attrib = train_X.shape
    #stopping criteria of the SGD loop
    stop_criteria = 2e-1
    #used to track if stop criteria has been hit
    stop_flag = False
    #flag to control checking during adaptive rate learning
    cost_prev = cost = 0
    tol_breach_counter=0
    total_tol_breach=0
    
    if non_linear_operation == 'sigmoid':
        print("will be using sigmoid as activation function")
    else:
        print("will be using relu as activation function")
        
    #if adaptive_learning_rate=='variable', need to revert the learning
    #whenever cost is greater than 'tol'
    #initialize the bias & weight vectors for the NN layers    
    prev_biases, prev_weights = init_params(layers_dims)

    
    for j in range(epochs):

        #debug code to profile training time, program execution time
        #currenttime = datetime.datetime.now()

        #shuffle data for each epoch
        train_X = np.append(train_X,train_Y,axis=1)
        np.random.shuffle(train_X)
        train_Y=train_X[:,n_attrib:]
        train_X=train_X[:,0:n_attrib]

        # run SGD for set of batch size
        for k in range(0, n_samples, mini_batch_size):
            # slide data for batch size
            x = train_X[k:k+mini_batch_size,:]
            y = train_Y[k:k+mini_batch_size,:]
            
            # run SGD algo on batch of samples
            biases, weights = batch_prop(x, y, non_linear_operation, \
                                                learn_rate, biases, weights)
        
        #compute cost as per learned parameters
        cost = compute_cost(non_linear_operation, train_X, train_Y, biases, weights)
        print( "Epoch {0}: Training.. Cost:{1}".format(j, cost))
        
        #for first iteration, do necessary initialization
        if j == 0:
            cost_prev = cost
            #learning param of previous iteration are stored for 'adaptive rate'
            for i in range(len(biases)):
                prev_biases[i] = biases[i]
                prev_weights[i] = weights[i]
            
        #check if stopping criteria has met & set flag
        if cost < stop_criteria:
            stop_flag = True
            break
        
        #handling for adaptive_learning_rate
        if j > 10 and adaptive_learning_rate == 'variable':
            if (cost_prev - cost > 1e-4):
                # we are doing well in learning, no need to keep track of 
                # previous learning parameters
                cost_prev = cost
                total_tol_breach = 0
                for i in range(len(biases)):
                    prev_biases[i] = biases[i]
                    prev_weights[i] = weights[i]
            #it means cost has increased more than tol=1e-4, divide learning rate by 5
            else:
                tol_breach_counter+=1
                total_tol_breach+=1
                if tol_breach_counter == 10:
                    learn_rate = learn_rate / 5
                    print("new learning rate", learn_rate)
                    tol_breach_counter = 0
                    
                    #forget current epoc & revert the parameters to previous ones
                    for i in range(len(biases)):
                        biases[i] = prev_biases[i]
                        weights[i] = prev_weights[i]
        
        if adaptive_learning_rate == 'variable':
            if total_tol_breach == 50 or learn_rate < 5e-05:
                #use the preserved parameters
                for i in range(len(biases)):
                    biases[i] = prev_biases[i]
                    weights[i] = prev_weights[i]
                break
            
        #debug code
        #print("Traing time of epoch", (datetime.datetime.now()-currenttime).seconds)
        
    #if stopping criteria did not meet, alert user to increase epoc freq
    if stop_flag:        
        print("Cost stopping criteria {0} hit ",format(stop_criteria))
    else:
        print("Stopping criteria not met. Increase epoc frequency & run again")
        
    print("final learning rate:",learn_rate)
    print("Cost:",cost)

    #do prediction over training data set        
    train_correct,train_y_hat, train_y = predict(non_linear_operation, train_X, train_Y, biases, weights)
    train_accuracy = train_correct/n_samples
    print("Training Set Accuracy:",train_accuracy)
    
    #do prediction over testing data set
    test_correct, test_y_hat, test_y = predict(non_linear_operation, test_X, test_Y, biases, weights)
    test_accuracy = test_correct/test_X.shape[0]
    print("Test Set Accuracy:",test_accuracy)
    
    #plot confusion matrix - training set
    plot_confusion_matrix('Training set :' + str(layers_dims), train_y_hat, train_y)
    
    #plot confusion matrix - test set
    plot_confusion_matrix('Test set :' + str(layers_dims), test_y_hat, test_y)
    
    #time.sleep( 15 )
    
    return train_accuracy, test_accuracy

"""
This function updates the network's weights and biases by applying
    gradient descent using backpropagation over the data after the  
    forward progagation cycle is passed

Input Parameters:
    train_X - training data
    train_Y - training labels
    non_linear_operation - could be 'sigmoid' or 'relu' for hidden units
    learn_rate - hyperparameter for learning rate that SGD uses
    biases, weights - learning parameters
Output:
    Returns: updated biases and weights
"""
def batch_prop(X, Y, non_linear_operation, learn_rate, biases, weights):

    #variables for Cost derivatives w.r.t bias & weight for whole batch
    db_batch=[]
    dw_batch=[]
    for i in range(len(biases)):
        db_batch.append(np.zeros(biases[i].shape))
        dw_batch.append(np.zeros(weights[i].shape))

    # loop to process samples of input batch of samples
    for pos in range(X.shape[0]):
        x=X[pos,:]
        y=Y[pos,:]
        x=np.reshape(x,(len(x),1))
        y=np.reshape(y,(len(y),1))

        #variables for Cost derivatives w.r.t bias & weight at sample level
        db_delta=[]
        dw_delta=[]
        for i in range(len(biases)):
            db_delta.append(np.zeros(biases[i].shape))
            dw_delta.append(np.zeros(weights[i].shape))
    
        ########################### forward prop ##########################
        # forward propagate for the batch samples
        activation = x
        # 'activations' is the list to store all the activations, layer by layer
        activations_list = [x]
        # list to store all the z vectors (each layer)
        zlist = [] 
        # forward propagate at each layer
        for i in range(len(biases)):
            # apply linear function
            z = np.dot(weights[i], activation)+biases[i]
            zlist.append(z)
            
            #if specified in config.txt, apply relu over hidden layer
            if i == len(biases)-1: # output layer
                activation = sigmoid(z)
            else:
                #hidden layers
                if non_linear_operation == 'sigmoid':
                    activation = sigmoid(z)
                else:
                    activation = relu(z)
            activations_list.append(activation)
    
        ########################### backward prop ##########################
        # back prop at output layer
        delta = (activations_list[-1] - y) * sigmoid_backward(zlist[-1])
        db_delta[-1] = delta
        dw_delta[-1] = np.dot(delta, activations_list[-2].transpose())
        
        # back prop - hidden layers
        for l in range(2, len(activations_list)):
            z = zlist[-l]
            #if specified in config.txt, handle for relu
            if non_linear_operation == 'sigmoid':
                sp = sigmoid_backward(z)
            else:
                sp = relu_backward(z)

            delta = np.dot(weights[-l+1].transpose(), delta) * sp
            db_delta[-l] = delta
            dw_delta[-l] = np.dot(delta, activations_list[-l-1].transpose())

        # add the derivates of this samples into the batch variables
        for i in range(len(db_batch)):
            db_batch[i] += db_delta[i]
            dw_batch[i] += dw_delta[i]
    
    # update parameters based on learning over the batch size
    for i in range(len(biases)):
        weights[i] = weights[i] - learn_rate * dw_batch[i]
        biases[i] = biases[i] - learn_rate * db_batch[i]
    
    return biases, weights

"""The sigmoid function."""
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

"""Derivative of the sigmoid function."""
def sigmoid_backward(z):
    return sigmoid(z)*(1-sigmoid(z))

"""The relu function."""
def relu(z):
    return np.maximum(0,z)

"""Derivative of the relu function."""
def relu_backward(z):
    # create object of size z.
    dz = np.array(z, copy=True)

    return np.maximum(0,dz)

"""
Implement the cost function (Mean square error) by forward propagating from 
input layer to the output layer & then calculates the error

Arguments:
    non_linear_operation - could be 'sigmoid' or 'relu' for hidden units
    a - input data (to start with, a is the 'x' data & then propagates the activation)
    biases, weights - learned parameters

Returns:
    a -- activation output of final layer
"""
def forward_prop(non_linear_operation, a, biases, weights):

    for b, w in zip(biases, weights):
        if non_linear_operation == 'sigmoid':
            a = sigmoid(np.dot(w, a)+b)
        else:
            a = relu(np.dot(w, a)+b)
    return a

"""
Implement the cost function (Mean square error) by forward propagating from 
input layer to the output layer & then calculates the error

Arguments:
    non_linear_operation - could be 'sigmoid' or 'relu' for hidden units
    x - training data
    y - training labels
    biases, weights - learned parameters

Returns:
    count -- count of correct predictions
    y_hat_lst - list of predicted output
"""
def predict(non_linear_operation, x, y, biases, weights):
    count=0
    y_hat_lst=[]
    y_lst=[]
    #predict for all samples 'x'
    for i in range(x.shape[0]):
        xx=np.reshape(x[i,:],(85,1))
        #output unit with maximum no. gives the predicted number
        y_hat=np.argmax(forward_prop(non_linear_operation, xx, biases, weights))
        yy=np.argmax(y[i,:])
        y_hat_lst.append(y_hat)
        y_lst.append(yy)
        count += (int(y_hat == yy))
    return count, y_hat_lst, y_lst


"""
Implement the cost function (Mean square error) by forward propagating from 
input layer to the output layer & then calculates the error

Arguments:
    non_linear_operation - could be 'sigmoid' or 'relu' for hidden units
    x - training data
    y - training labels
    biases - list of dimensions of activation units at each layer
                e.g [85 30 10] architecture is like In-Hidden-Out as 85*30*10
    weights - epoch count (no. of iterations for which SGD will run)

Returns:
    cost -- Mean square error
"""
def compute_cost(non_linear_operation, x, y, biases, weights):
    
    cost=0
    n_samples=x.shape[0]

    #calculate cost of all samples    
    for i in range(x.shape[0]):
        xx=np.reshape(x[i,:],(85,1))
        #forward propagate to get the prediction
        y_hat = np.argmax(forward_prop(non_linear_operation, xx, biases, weights))
        yy = np.argmax(y[i,:])
        
        #cost += np.sum(np.dot( (y_hat-yy), (y_hat-yy).T ))
        cost += np.square(y_hat-yy)
    
    #returns mean of cost
    return cost/n_samples


# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:28:57 2019

@author: abhishekt
"""


"""
   plot_confusion_matrix: Plot the confusion matrix
   
   Input:  pred_rating - list of predictions
           actual_rating - list of actual target values
   Returns: None
"""
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

def plot_confusion_matrix(title, pred_rating, actual_rating):
    #class_names
    classes = ['0','1','2','3','4','5','6','7','8','9']
    
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    cnf_matrix = confusion_matrix(np.asarray(actual_rating), np.asarray(pred_rating))
    #print(cnf_matrix)
    plt.figure(figsize = (10,6))
    sn.heatmap(cnf_matrix, annot=True)
    
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=90)
    
    plt.show()

"""
   plot_Accuracy: Function to plot Validation & Test Accuracy
"""
def plot_Accuracy(num_hidden_units, train_accu, test_accu, train_label, test_label):

    #create figure object for plotting
    plt.figure(figsize=(7, 5))
    
    #plot accuracy on Y-axis & log(C) on x-axis
    plt.plot(num_hidden_units, train_accu, color = 'blue', label = train_label)
    plt.plot(num_hidden_units, test_accu, color = 'green', label = test_label)
    plt.title('Train & Test Accuracy Plot')
    plt.xlabel("Number of hidden units (constant for each hidden layer) -->")
    plt.ylabel("Prediction Accuracies")
    plt.legend(loc='lower right', shadow=True, fontsize='large', numpoints=1)
    plt.show()

def relu(Z):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


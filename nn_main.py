# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:56:41 2019

@author: abhishekt
"""

import sys
import pickle
import numpy as np
import data_handler
from neural_lib import *

option = sys.argv[1]

if option == 'a':
    print("in option 'a'")
    train_file = sys.argv[2] # training data-set
    test_file = sys.argv[3]  # test data-set
    one_hot_enc_train_file = sys.argv[4] #one-hot encoded training data
    one_hot_enc_test_file = sys.argv[5]  #one-hot encoded test data
    
    # dump_data is used to save the hot-encoded training & test data into pickle
    # files. When 1, saves data.   When 0, reads the data from pickle file &
    # saves the data processing time each time program is run.
    dump_data = int(sys.argv[6])
    if dump_data == 1:
        #one-hot-encode data of training & test files specified
        df_train, df_test = data_handler.one_hot_encode_data(train_file, test_file)
        print("Training data dimension:",df_train.shape)
        print("Test data dimension:",df_test.shape)
        
        #save training data into pickle file
        with open(one_hot_enc_train_file, 'wb+') as f:
                    pickle.dump(df_train, f, pickle.HIGHEST_PROTOCOL)
    
        #save test data into pickle file
        with open(one_hot_enc_test_file, 'wb+') as f:
                    pickle.dump(df_test, f, pickle.HIGHEST_PROTOCOL)
        
        print("One hot encoded data saved")

    else:
        # loads training & test data from pickle files
        with open(one_hot_enc_train_file, 'rb') as f:
            df_train = pickle.load(f)
            print("Training data dimension:",df_train.shape)

        with open(one_hot_enc_test_file, 'rb') as f:
            df_test = pickle.load(f)
            print("Test data dimension:",df_test.shape)


if option == 'b':
    print("in option 'b'")
    #takes 4 arguments - sample given below
    #run nn_main.py b config.txt poker-hand-training-true.data poker-hand-testing.data 5
    config_file = sys.argv[2] #config.txt
    train_file = sys.argv[3] # training data-set
    test_file = sys.argv[4]  # test data-set
    # variable used to run NN model in loop with different configuations
    temp_loop_size = int(sys.argv[5]) 
    
    with open('config.txt') as f:
        content = f.readlines()
    # remove `\n` at the end of each line
    lines = [line.rstrip('\n') for line in content]
    
    input_size = int(lines[0])
    #print(type(input_size))
    output_size = int(lines[1])
    batch_size = int(lines[2])
    hidden_layers = int(lines[3])
    hidden_units = (np.asarray(lines[4].split(" "))).astype(int)
    #print("hidden_units[0]",hidden_units[0])
    #print("hidden_units[1]",hidden_units[1])
    non_linear_operation = lines[5]
    #print(type(non_linear_operation))
    adaptive_learning_rate = lines[6]
    #print(type(adaptive_learning_rate))
    #print(adaptive_learning_rate)

    df_train_X, df_train_Y, df_test_X, df_test_Y = data_handler.get_data\
                                                (train_file, test_file)
    
    #this list is used during graph plot - Number of Units on X-axis
    Num_hidden_units=[]
    #this list is used during graph plot - Training accuracy on Y-axis
    train_accuracy_list=[]
    #this list is used during graph plot - test accuracy on Y-axis
    test_accuracy_list=[]
    
    for i in range(1,temp_loop_size+1):
        
        #this list is used to contruct the NN architecture 
        # contains no. of units in each layer => (Input-hidden-Output)
        layers_dims=[]
        layers_dims.append(input_size)
        [layers_dims.append(hidden*i) for hidden in hidden_units]
        layers_dims.append(output_size)
        
        Num_hidden_units.append(hidden_units[0]*i)
        
        epoch = 100
        learn_rate = 0.1
        
        #build NN model as per specified parameters in config.txt
        train_accuracy, test_accuracy = generic_NN_model(df_train_X, df_train_Y,\
                    df_test_X, df_test_Y, layers_dims, epoch, batch_size, \
                    non_linear_operation, learn_rate, adaptive_learning_rate)
        
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        
    #plot graph - Hidden units vs Accuracy
    train_label = 'Training Set Accuracy'
    test_label = 'Test Set Accuracy'    
    plot_Accuracy(Num_hidden_units, train_accuracy_list, test_accuracy_list, train_label, test_label)

if option == 'd':
    print("in option 'c'")
    #takes 4 arguments - sample given below
    #run nn_main.py b config.txt poker-hand-training-true.data poker-hand-testing.data 5
    config_file = sys.argv[2]
    temp_loop_size = int(sys.argv[3])

    with open('config.txt') as f:
        content = f.readlines()
    # remove `\n` at the end of each line
    lines = [line.rstrip('\n') for line in content]
    
    hidden_layers = int(lines[3])
    hidden_units = (np.asarray(lines[4].split(" "))).astype(int)

    for i in range(1,temp_loop_size+1):
        #this list is used during graph plot - Number of Units on X-axis
        Num_hidden_units=[]
        #this list is used during graph plot - Training accuracy on Y-axis
        train_accuracy_list=[]
        #this list is used during graph plot - test accuracy on Y-axis
        test_accuracy_list=[]

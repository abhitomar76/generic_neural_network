# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:11:36 2019

@author: abhishekt
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

"""
This function one-hot encodes all data columns except the label & returns
train & test data
"""
def one_hot_encode_data(train_file, test_file):

    #load training data
    #df_train = pd.read_csv('poker-hand-training-true.data', sep=",", header = None)
    df_train = pd.read_csv(train_file, sep=",", header = None)
    m_train, n_train = df_train.shape
    #print(df_train.shape)

    #load testing data
    #df_test = pd.read_csv('poker-hand-testing.data', sep=",", header = None)
    df_test = pd.read_csv(test_file, sep=",", header = None)
    m_test, n_test = df_test.shape
    #print(df_test.shape)

    #append test data to training data (else number of one-hot-encode features 
    #in both sets may vary resulting in issues later)
    df_train=df_train.append(df_test)
    #print(df_train.shape)
    
    ################### apply one-hot encoding #########################
    #apply one-hot encoding to all columns except last one which is 'label'
    column_trans = ColumnTransformer([('UCI-Poker', OneHotEncoder\
                (categories='auto',dtype='int8'),[0,1,2,3,4,5,6,7,8,9]),]\
                , remainder='passthrough')
    
    #fit training data & transform to one-hot-encode    
    column_train_trans=column_trans.fit(df_train)
    train_transform = column_train_trans.transform(df_train).toarray()
    
    #print("transformed train data")
    
    del df_train
    del df_test

    #now segregate the training & test data from one-hot transformed output    
    df_train = train_transform[0:m_train,:]
    df_test =  train_transform[m_train:,:]
    
    del train_transform
    
    return (df_train, df_test)

"""
This function one-hot encodes all data columns & returns
train & test data (attributes & label separately)
"""
def get_data(train_file, test_file):

    #df_train = pd.read_csv('poker-hand-training-true.data', sep=",", header = None)
    df_train = pd.read_csv(train_file, sep=",", header = None)
    m_train, n_train = df_train.shape
    print(df_train.shape)

    #df_test = pd.read_csv('poker-hand-testing.data', sep=",", header = None)
    df_test = pd.read_csv(test_file, sep=",", header = None)
    m_test, n_test = df_test.shape
    print(df_test.shape)

################### apply one-hot encoding #########################
    #append test data to training data (else number of one-hot-encode features 
    #in both sets may vary resulting in issues later)
    df_train=df_train.append(df_test)
    print(df_train.shape)
    
    #apply one-hot encoding to all columns
    column_trans = ColumnTransformer([('UCI-Poker', OneHotEncoder(categories='auto',dtype='int8'),[0,1,2,3,4,5,6,7,8,9,10]),], remainder='passthrough')
    
    #fit training data & transform to one-hot-encode    
    column_train_trans=column_trans.fit(df_train)
    train_transform = column_train_trans.transform(df_train).toarray()
    #train_transform = column_train_trans.transform(df_train)
    
    print("transformed train data")
    #print(type(train_transform))
    #print(train_transform.shape)
    
    #update variable for no. of attributes (as it might have changed after\
                                                        #transform)
    n_train = train_transform.shape[1]
    #print(train_transform[:,n_train-1])
    
    del df_train
    del df_test
    
    #segragate train data set
    df_train_X = train_transform[0:m_train,0:n_train-10]
    df_train_Y = train_transform[0:m_train,n_train-10:]

    print("df_train_X.shape",df_train_X.shape)
    print("df_train_Y.shape",df_train_Y.shape)
    
    #segragate test data set
    df_test_X =  train_transform[m_train:,0:n_train-10]
    df_test_Y =  train_transform[m_train:,n_train-10:]
    print("df_test_X.shape",df_test_X.shape)
    print("df_test_Y.shape",df_test_Y.shape)
    
    del train_transform
    
    return (df_train_X, df_train_Y, df_test_X, df_test_Y)

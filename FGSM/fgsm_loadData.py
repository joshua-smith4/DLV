#!/usr/bin/env python

"""
The main file for load models 

author: Xiaowei Huang
"""

import sys


import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pylab import *

# keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense
import keras.optimizers


# visualisation
#from keras.utils.visualize_util import plot
#
from keras.datasets import mnist
from keras.utils import np_utils

# for training cifar10
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint



from configuration import *



def fgsm_loadData(model): 

    if dataset == "mnist": 
        (X_train, Y_train, X_test, Y_test, batch_size, nb_epoch) = NN.read_dataset()
        
    elif dataset == "gtsrb":     
        X_train, Y_train = dataBasics.read_dataset()
        #X_train = X_train[:1000]
        #Y_train = Y_train[:1000]
        
        X_test = X_train
        Y_test = Y_train

        
    elif dataset == "cifar10": 
        (X_train,Y_train,X_test,Y_test, img_channels, img_rows, img_cols, batch_size, nb_classes, nb_epoch, data_augmentation) = NN.read_dataset()

        
    elif dataset == "imageNet": 
        print "we do not have the set of training data for imageNet"
        X_train, Y_train = np.zeros((1,3,224,224)), np.zero((1,))
        
        X_test = X_train
        Y_test = Y_train
        
    elif dataset == "twoDcurve": 
        # define and construct model

        # load data
        N_samples = 5000
        N_tests = 1000
        X_train, Y_train, X_test, Y_test = NN.load_data(N_samples,N_tests)
        
    Y_predicted = model.predict(X_train)
    
    return X_train, Y_train, Y_predicted, X_test, Y_test


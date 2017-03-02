#!/usr/bin/env python

"""
fgsm main file

author: Xiaowei Huang
"""

import sys


import time
import numpy as np
import copy 
import random
import matplotlib.pyplot as plt

from loadData import loadData 

from configuration import *
from basics import *
from networkBasics import *

from fgsm_loadData import fgsm_loadData
from attacks_th import fgsm
from utils_th import batch_eval

import theano
import theano.tensor as T
        
def fgsm_main(model,eps):

    # FGSM adversary examples
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', default=128, help='Size of training batches')
    parser.add_argument('--train_dir', '-d', default='/tmp', help='Directory storing the saved model.')
    parser.add_argument('--filename', '-f',  default='mnist.ckpt', help='Filename to save model under.')
    parser.add_argument('--nb_epochs', '-e', default=6, type=int, help='Number of epochs to train model')
    parser.add_argument('--learning_rate', '-lr', default=0.5, type=float, help='Learning rate for training')
    args = parser.parse_args()
    
    x_train, y_train, y_predicted = fgsm_loadData(model)
    x_shape = x_train.shape
    model.build(x_shape)
    
    x = T.tensor4('x')
    predictions = model(x)
    
    adv_x = fgsm(x,predictions,eps)
    X_test_adv, = batch_eval([x], [adv_x], [x_train], args=args)

    print X_test_adv.shape
    y_predicted2 = model.predict(X_test_adv)

    nd = 0 
    for i in range(len(y_predicted)): 
        if np.argmax(y_predicted[i]) != np.argmax(y_predicted2[i]): 
            nd += 1
    print "%s diff in %s examples"%(nd,len(y_predicted))    

    return nd
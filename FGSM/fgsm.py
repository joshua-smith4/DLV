#!/usr/bin/env python

"""
fgsm main file
adapted from 

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
from fgsm_dataCollection import *
from attacks_th import fgsm
from utils_th import batch_eval

import theano
import theano.tensor as T
        
def fgsm_main(model,eps):

    dc = fgsm_dataCollection()


    # FGSM adversary examples
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', default=128, help='Size of training batches')
    parser.add_argument('--train_dir', '-d', default='/tmp', help='Directory storing the saved model.')
    parser.add_argument('--filename', '-f',  default='mnist.ckpt', help='Filename to save model under.')
    parser.add_argument('--nb_epochs', '-e', default=6, type=int, help='Number of epochs to train model')
    parser.add_argument('--learning_rate', '-lr', default=0.5, type=float, help='Learning rate for training')
    args = parser.parse_args()
    
    x_train, y_train, y_predicted, x_test, y_test = fgsm_loadData(model)
    
    x_train = x_test
    y_train = y_test
    y_predicted = model.predict(x_train)
    imageIndex = 4357
    filename = "/Users/xiaowei/Repositories/DLV/FGSM/pic/%s_fgsm.jpg"%(imageIndex)
    testImage = np.squeeze(x_train[imageIndex])
    save(0,testImage,filename)
    
    x_shape = x_train.shape
    model.build(x_shape)
    
    x = T.tensor4('x')
    predictions = model(x)
    
    adv_x = fgsm(x,predictions,eps)
    x_train_adv, = batch_eval([x], [adv_x], [x_train], args=args)
    

    print x_train_adv.shape
    y_predicted_adv = model.predict(x_train_adv)

    nd = 0 
    sumOfeuD = 0 
    sumOfl1D = 0
    for i in range(len(y_predicted)): 
        if np.argmax(y_predicted[i]) != np.argmax(y_predicted_adv[i]): 
            nd += 1
            sumOfeuD += euclideanDistance(x_train[i],x_train_adv[i])
            sumOfl1D += l1Distance(x_train[i],x_train_adv[i])
    print "%s diff in %s examples"%(nd,len(y_predicted))  
    
    conf = max(y_predicted_adv[imageIndex])
    cls = np.argmax(y_predicted_adv[imageIndex])
    filename = "/Users/xiaowei/Repositories/DLV/FGSM/pic/%s_fgsm_%s_%s_%s.jpg"%(imageIndex,eps,cls,conf)                        
    testImage1 = np.squeeze(x_train_adv[imageIndex])
    print("euclidean distance: %s"%(euclideanDistance(testImage1,testImage))) 
    print("L1 distance: %s"%(l1Distance(testImage1,testImage)))
    save(0,testImage1,filename)

    
    # calculate the average Euclidean distance of the diff examples
    eud = sumOfeuD / nd
    l1d = sumOfl1D / nd
    print "in %s diff examples, the average eclidean distance is %s"%(nd,eud)  
    print "in %s diff examples, the average L1 distance is %s"%(nd,l1d)  
    
    dc.updateNDiffs(nd)    
    dc.updateEps(eps)
    dc.updateEuclideanDistance(eud)
    dc.updatel1Distance(l1d)
    dc.summarise()
    dc.close()

    return nd
    
def save(layer,image,filename):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image * 255, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.savefig(filename)

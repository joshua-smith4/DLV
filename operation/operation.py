#!/usr/bin/env python

"""
author: Xiaowei Huang

operation on the NN model
"""



from basics import *
from networkBasics import *
from configuration import * 




def cuttingModel(model,layerToCut,image):
    
    print "start cutting model"
    layerToCut = 2
    (origclass,origconf) = NN.predictWithImage(model,image)
    cutactivations = NN.getActivationValue(model,layerToCut-1,image)
    cutModel = NN.dynamic_build_model(layerToCut,cutactivations.shape)
    cutModel = NN.dynamic_read_model_from_file(cutModel,'%s/mnist.mat'%directory_model_string,'%s/mnist.json'%directory_model_string,layerToCut)
    (cutclass,cutconf) = NN.predictWithImage(cutModel,cutactivations)
    print "on original model: %s (class) %s (confidence)"%(origclass,origconf)
    print "on cutted model: %s (class) %s (confidence)"%(cutclass,cutconf)

    print "finish cutting model"
    
    return cutModel,cutactivations
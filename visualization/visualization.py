#!/usr/bin/env python

"""
author: Xiaowei Huang

Visualizaiton operations
"""

import numpy as np
import time
import copy
import math
import random
from operator import mul

import matplotlib.pyplot as plt
from scipy import ndimage

from basics import *
from networkBasics import *
from configuration import * 

from imageProcessing import mergeImagesHorizontal, mergeImagesVertical


def visualization(model,dataset,imageIndex,originalImage):
    
    print "start visualization ... "
    dataBasics.save(0,originalImage,"%s/image_%s.png"%(directory_pic_string,imageIndex))
    #print "please refer to the file "+directory_pic_string+"/temp.png for the image under processing ..."
    
    config = NN.getConfig(model)
    
    imageNames = []
    for layer2Consider in range(len(config)):
        imageOneLayer = visualizeOneLayer(model,originalImage,layer2Consider)
        if imageOneLayer != "":
            imageNames.append(imageOneLayer)
    mergeImagesHorizontal(imageNames,"%s/image_%s_merged.png"%(directory_pic_string,imageIndex))
        
    print "end of visualization ... "
    
    
def visualizeOneLayer(model,originalImage,layer2Consider):
    
        
    # get weights and bias of the entire trained neural network
    (wv,bv) = NN.getWeightVector(model,layer2Consider)
        
    activations = NN.getActivationValue(model,layer2Consider,originalImage)
    layerType = getLayerType(model,layer2Consider)
    wv2Consider, bv2Consider = getWeight(wv,bv,layer2Consider)

    imageNames = []
    for nodeIndex in range(len(activations)): 
        if len(activations.shape) >= 3: 
            if dataset == "mnist": 
                dataBasics.save(0,activations[nodeIndex],"%s/temp_%s_%s.png"%(directory_pic_string,layer2Consider,nodeIndex))
            else: 
                dataBasics.save(0,activations[nodeIndex:nodeIndex+1],"%s/temp_%s_%s.png"%(directory_pic_string,layer2Consider,nodeIndex))
            #print "please refer to the file %s/temp_%s_%s.png for the image under processing ..."%(directory_pic_string,layer2Consider,nodeIndex)
            imageNames.append("%s/temp_%s_%s.png"%(directory_pic_string,layer2Consider,nodeIndex)) 
        #else: 
            #print "not an image for layer %s"%(layer2Consider)
        
    if len(imageNames) != 0: 
        mergeImagesVertical(imageNames,"%s/test_%s.png"%(directory_pic_string,layer2Consider))
        return ("%s/test_%s.png"%(directory_pic_string,layer2Consider))
    else: 
        return ""


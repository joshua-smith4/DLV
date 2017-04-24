#!/usr/bin/env python

"""
Define paramters
author: Xiaowei Huang
"""

from network_configuration import *
from usual_configuration import * 


#######################################################
#
#  The following are parameters to indicate how to work 
#   with a problem
#
#######################################################

# which dataset to work with
#dataset = "twoDcurve"
dataset = "mnist"
#dataset = "gtsrb"
#dataset = "cifar10"
#dataset = "imageNet"

# the network is trained from scratch
#  or read from the saved files
whichMode = "read"
#whichMode = "train"

# work with a single image or a batch of images 
#dataProcessing = "single"
dataProcessing = "batch"
dataProcessingBatchNum = 1


#######################################################
#
#  1. parameters related to the networks
#
#######################################################


(featureDims,span,numSpan,errorBounds,boundOfPixelValue,NN,dataBasics,directory_model_string,directory_statistics_string,directory_pic_string,filterSize) = network_parameters(dataset)


#######################################################
#
#  2. parameters related to the experiments
#
#######################################################


(startIndexOfImage,startLayer, maxLayer,numOfFeatures,controlledSearch,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,enumerationMethod,checkingMode,exitWhen) = usual_configuration(dataset)
    

############################################################
#
#  3. other parameters that are believed to be shared among all cases
#  FIXME: check to see if they are really needed/used
#
################################################################



# timeout for z3 to handle a run
timeout = 600



############################################################
#
#  some miscellaneous parameters 
#   which need to confirm whether they are useful
#  FIXME: check to see if they are really needed/used
#
################################################################


# the error bound for manipulation refinement 
# between layers
epsilon = 0.1


############################################################
#
#  a parameter to decide whether 
#  FIXME: check to see if they are really needed/used
#
################################################################

# 1) the stretch is to decide a dimension of the next layer on 
#     the entire region of the current layer
# 2) the condense is to decide several (depends on refinementRate) dimensions 
#     of the next layer on a manipulation of a single dimension of the current layer

#regionSynthMethod = "stretch"
regionSynthMethod = "condense"

    
#######################################################
#
#  show detailedInformation or not
#  FIXME: check to see if they are really needed/used
#
#######################################################

def nprint(str):
    return      
        

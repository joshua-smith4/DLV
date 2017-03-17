#!/usr/bin/env python

"""
author: Xiaowei Huang
"""

import numpy as np
import time
import os
import copy

from configuration import directory_statistics_string

class fgsm_dataCollection:

    fileName = "%s/fgsm_dataCollection.txt"%(directory_statistics_string)
    fileHandler = 0

    def __init__(self):
        self.eps = 0
        self.euclideanDistance = 0
        self.l1Distance = 0
        self.fileHandler = open(self.fileName, 'a')
        self.ndiffs = 0
        
    def updateEps(self,eps):
        self.eps = eps
        
    def updateNDiffs(self,ndiffs):
        self.ndiffs = ndiffs
        
    def updateEuclideanDistance(self,eud):
        self.euclideanDistance = eud
        
    def updatel1Distance(self,l1d):
        self.l1Distance = l1d        
            
    def summarise(self):
        self.fileHandler.write("variant: %s\n"%(self.eps))
        self.fileHandler.write("number of differences: %s\n"%(self.ndiffs))
        self.fileHandler.write("average euclidean distance: %s\n"%(self.euclideanDistance))
        self.fileHandler.write("average L1 distance: %s\n"%(self.l1Distance))

    def close(self):
        self.fileHandler.close()
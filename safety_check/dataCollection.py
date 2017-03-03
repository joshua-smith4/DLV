#!/usr/bin/env python

"""
author: Xiaowei Huang
"""

import numpy as np
import time
import os
import copy

from configuration import directory_statistics_string

class dataCollection:

    index = 0
    layer = 0
    fileName = "%s/dataCollection.txt"%(directory_statistics_string)
    fileHandler = 0

    def __init__(self):
        self.runningTime = {}
        self.manipulationPercentage = {}
        self.euclideanDistance = {}
        self.l1Distance = {}
        self.confidence = {}
        self.fileHandler = open(self.fileName, 'w')
        
    def initialiseIndex(self, index):
        self.index = index
        
    def initialiseLayer(self, layer):
        self.layer = layer
        
    def addRunningTime(self, rt):
        self.runningTime[self.index,self.layer] = rt
        
    def addConfidence(self, cf):
        self.confidence[self.index,self.layer] = cf
        
    def addManipulationPercentage(self, mp):
        self.manipulationPercentage[self.index,self.layer] = mp
        
    def addEuclideanDistance(self, eudist):
        self.euclideanDistance[self.index,self.layer] = eudist
        
    def addl1Distance(self, l1dist):
        self.l1Distance[self.index,self.layer] = l1dist

    def provideDetails(self): 
        self.fileHandler.write("running time: \n")
        for i,r in self.runningTime.iteritems():
            self.fileHandler.write("%s:%s\n"%(i,r))
            
        self.fileHandler.write("manipulation percentage: \n")
        for i,r in self.manipulationPercentage.iteritems():
            self.fileHandler.write("%s:%s\n"%(i,r))
            
        self.fileHandler.write("Euclidean distance: \n")
        for i,r in self.euclideanDistance.iteritems():
            self.fileHandler.write("%s:%s\n"%(i,r))
            
        self.fileHandler.write("L1 distance: \n")
        for i,r in self.l1Distance.iteritems():
            self.fileHandler.write("%s:%s\n"%(i,r))
            
        self.fileHandler.write("confidence: \n")
        for i,r in self.confidence.iteritems():
            self.fileHandler.write("%s:%s\n"%(i,r))
        self.fileHandler.write("\n")
            
    def summarise(self):
        art = sum(self.runningTime.values()) / len(self.runningTime.values()) 
        self.fileHandler.write("average running time: %s\n"%(art))
        amp = sum(self.manipulationPercentage.values()) / len(self.manipulationPercentage.values())
        self.fileHandler.write("average manipulation percentage: %s\n"%(amp))
        eudist = sum(self.euclideanDistance.values()) / len(self.euclideanDistance.values())
        self.fileHandler.write("average euclidean distance: %s\n"%(eudist))
        l1dist = sum(self.l1Distance.values()) / len(self.l1Distance.values())
        self.fileHandler.write("average L1 distance: %s\n"%(l1dist))

    def close(self):
        self.fileHandler.close()
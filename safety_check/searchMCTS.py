#!/usr/bin/env python

"""
A data structure for organising search

author: Xiaowei Huang
"""

import numpy as np
import time
import os
import copy
import sys
import operator
import random
import math

from configuration import *
from regionSynth import initialiseRegion, initialiseRegions

from inputManipulation import applyManipulation
from basics import mergeTwoDicts, diffPercent, euclideanDistance, l1Distance, numDiffs

cp = 0.5


class searchMCTS:

    def __init__(self, model, image, k):
        self.image = image
        self.model = model
        
        self.spans = {}
        self.numSpans = {}
        self.cost = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}
        self.numberOfVisited = {}
        
        self.indexToNow = 0
        # the layer currently working on 
        self.maxilayer = k 
        # current root node
        self.rootIndex = 0
        
        # initialise root node
        self.spans[-1] = {}
        self.numSpans[-1] = {} 
        self.initialiseLeafNode(0,-1,[],[])
        
        (self.originalClass,self.originalConfident) = NN.predictWithImage(self.model,image)

        
    def initialiseLeafNode(self,index,parentIndex,newSpans,newNumSpans):
        print("initialising a leaf node %s from the node %s"%(index,parentIndex))
        self.spans[index] = mergeTwoDicts(self.spans[parentIndex],newSpans)
        self.numSpans[index] = mergeTwoDicts(self.numSpans[parentIndex],newNumSpans)
        self.cost[index] = 0
        self.parent[index] = parentIndex 
        self.children[index] = []
        self.fullyExpanded[index] = False
        self.numberOfVisited[index] = 0    
        

    def destructor(self): 
        self.image = 0
        self.model = 0
        self.spans = {}
        self.numSpans = {}
        self.cost = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}
        self.numberOfVisited = {}
        
    # move one step forward
    # it means that we need to remove children other than the new root
    def makeOneMove(self,newRootIndex): 
        print "making a move into the new root %s, whose value is %s and visited number is %s"%(newRootIndex,self.cost[newRootIndex],self.numberOfVisited[newRootIndex])
        self.removeChildren(self.rootIndex,[newRootIndex])
        self.rootIndex = newRootIndex
    
    def removeChildren(self,index,indicesToAvoid): 
        if self.fullyExpanded[index] == True: 
            for childIndex in self.children[index]: 
                if childIndex not in indicesToAvoid: self.removeChildren(childIndex,[])
        self.spans.pop(index,None)
        self.numSpans.pop(index,None)
        self.cost.pop(index,None) 
        self.parent.pop(index,None) 
        self.children.pop(index,None) 
        self.fullyExpanded.pop(index,None)
        self.numberOfVisited.pop(index,None)
            
    
    def bestChild(self,index):
        allValues = {}
        for childIndex in self.children[index]: 
            allValues[childIndex] = self.cost[childIndex]
        print("finding best children from %s"%(allValues))
        return max(allValues.iteritems(), key=operator.itemgetter(1))[0]
        
    def treeTraversal(self,index):
        if self.fullyExpanded[index] == True: 
            print("tree traversal on node %s"%(index))
            allValues = {}
            for childIndex in self.children[index]: 
                allValues[childIndex] = (self.cost[childIndex] / float(self.numberOfVisited[childIndex])) + cp * math.sqrt(math.log(self.numberOfVisited[index]) / float(self.numberOfVisited[childIndex]))
            nextIndex = max(allValues.iteritems(), key=operator.itemgetter(1))[0]
            return self.treeTraversal(nextIndex)
        else: 
            print("tree traversal terminated on node %s"%(index))
            return index
        
    def initialiseExplorationNode(self,index):
        print("expanding %s"%(index))
        for (span,numSpan,_) in initialiseRegions(self.model,self.image,self.spans[index].keys()): 
            self.indexToNow += 1
            self.initialiseLeafNode(self.indexToNow,index,span,numSpan)
            self.children[index].append(self.indexToNow)
        self.fullyExpanded[index] = True
        return self.children[index]

    def backPropagation(self,index,value): 
        self.cost[index] += value
        self.numberOfVisited[index] += 1
        if self.parent[index] in self.parent : 
            print("start backPropagating the value %s from node %s, whose parent node is %s"%(value,index,self.parent[index]))
            self.backPropagation(self.parent[index],value)
        else: 
            print("backPropagating ends on node %s"%(index))
            
    # start random sampling and return the eclidean value as the value
    def sampling(self,index):
        print("start sampling node %s"%(index))
        return self.sampleNext(self.spans[index],self.numSpans[index])
        #allChildren = initialiseRegions(model,self.image,self.spans[index].keys()) 
    
    def sampleNext(self,spansPath,numSpansPath): 
        #print spansPath.keys()
        image1 = applyManipulation(self.image,spansPath,numSpansPath)
        (newClass,newConfident) = NN.predictWithImage(self.model,image1)
        #print euclideanDistance(self.image,image1), newConfident, newClass
        (distMethod,distVal) = controlledSearch
        if distMethod == "euclidean": 
            dist = 1 - euclideanDistance(image1,self.image) 
            termValue = 0.0
        elif distMethod == "L1": 
            dist = 1 - l1Distance(image1,self.image) 
            termValue = 0.0
        elif distMethod == "Percentage": 
            dist = 1 - diffPercent(image1,self.image)
            termValue = 0.0
        elif distMethod == "NumDiffs": 
            dist = 1 - diffPercent(image1,self.image)
            termValue = 0.0
        termByDist = dist < 1 - distVal

        if newClass != self.originalClass: 
            return dist
        elif termByDist == True: 
            return termValue
        else: 
            #print("continue sampling node ... ")
            allChildren = initialiseRegions(self.model,self.image,spansPath.keys())
            randomIndex = random.randint(0, len(allChildren)-1)
            (span,numSpan,_) = allChildren[randomIndex]
            newSpanPath = self.mergeSpan(spansPath,span)
            newNumSpanPath = self.mergeNumSpan(numSpansPath,numSpan)
            return self.sampleNext(newSpanPath,newNumSpanPath)
            
    def terminalNode(self,index): 
        image1 = applyManipulation(self.image,self.spans[index],self.numSpans[index])
        (newClass,_) = NN.predictWithImage(self.model,image1)
        return newClass != self.originalClass 
        
    def euclideanDist(self,index): 
        image1 = applyManipulation(self.image,self.spans[index],self.numSpans[index])
        return euclideanDistance(self.image,image1)
        
    def l1Dist(self,index): 
        image1 = applyManipulation(self.image,self.spans[index],self.numSpans[index])
        return l1Distance(self.image,image1)
        
    def diffPercent(self,index): 
        image1 = applyManipulation(self.image,self.spans[index],self.numSpans[index])
        return diffPercent(self.image,image1)

    def mergeSpan(self,spansPath,span): 
        return mergeTwoDicts(spansPath, span)
        
    def mergeNumSpan(self,numSpansPath,numSpan):
        return mergeTwoDicts(numSpansPath, numSpan)
        
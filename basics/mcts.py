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
from basics import mergeTwoDicts, diffPercent, euclideanDistance, l1Distance, numDiffs, diffImage

from decisionTree import decisionTree
from re_training import re_training

# tunable parameter for MCTS
cp = 0.5


class mcts:

    def __init__(self, model, autoencoder, image, activations, layer):
        self.image = image
        self.activations = activations
        self.model = model
        self.autoencoder = autoencoder
        
        self.spans = {}
        self.numSpans = {}
        self.cost = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}
        self.numberOfVisited = {}
        
        self.indexToNow = 0
        # current root node
        self.rootIndex = 0
     
        # current layer
        self.layer = layer
        
        # initialise root node
        self.spans[-1] = {}
        self.numSpans[-1] = {}
        self.initialiseLeafNode(0,-1,[],[])
        
        # local actions
        self.actions = {}
        self.usedActionsID = {}
        self.indexToActionID = {}

        # best case
        self.bestCase = (0,{},{})
        
        # useless points
        self.uselessPixels = []
        
        (self.originalClass,self.originalConfident) = self.predictWithActivations(self.activations)
        
        self.decisionTree = 0
        self.re_training = re_training(model,self.image.shape)

    def predictWithActivations(self,activations):
        if self.layer > -1: 
            output = np.squeeze(self.autoencoder.predict(np.expand_dims(activations,axis=0)))
            return NN.predictWithImage(self.model,output)
        else: 
            return NN.predictWithImage(self.model,activations)
            
    def visualizationMCTS(self):
        for k in range(len(self.activations)): 
            activations1 = copy.deepcopy(self.activations)
            # use a random node to replace the feature node
            emptyNode = np.zeros_like(self.activations[0])
            activations1[k] = emptyNode
            output = np.squeeze(self.autoencoder.predict(np.expand_dims(activations1,axis=0)))
            path0="%s/%s_autoencoder_%s.png"%(directory_pic_string,startIndexOfImage,k)
            dataBasics.save(-1,output, path0)
        
    def initialiseActions(self): 
        allChildren = initialiseRegions(self.autoencoder,self.activations,[])
        for i in range(len(allChildren)): 
            self.actions[i] = allChildren[i] 
        print "%s actions have been initialised. "%(len(self.actions))
        # initialise decision tree
        self.decisionTree = decisionTree(self.actions)
        
    def initialiseLeafNode(self,index,parentIndex,newSpans,newNumSpans):
        nprint("initialising a leaf node %s from the node %s"%(index,parentIndex))
        self.spans[index] = mergeTwoDicts(self.spans[parentIndex],newSpans)
        self.numSpans[index] = mergeTwoDicts(self.numSpans[parentIndex],newNumSpans)
        self.cost[index] = 0
        self.parent[index] = parentIndex 
        self.children[index] = []
        self.fullyExpanded[index] = False
        self.numberOfVisited[index] = 0    

    def destructor(self): 
        self.image = 0
        self.activations = 0
        self.model = 0
        self.autoencoder = 0
        self.spans = {}
        self.numSpans = {}
        self.cost = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}
        self.numberOfVisited = {}
        
        self.actions = {}
        self.usedActionsID = {}
        self.indexToActionID = {}
        
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
            nprint("tree traversal on node %s"%(index))
            allValues = {}
            for childIndex in self.children[index]: 
                allValues[childIndex] = (self.cost[childIndex] / float(self.numberOfVisited[childIndex])) + cp * math.sqrt(math.log(self.numberOfVisited[index]) / float(self.numberOfVisited[childIndex]))
            nextIndex = max(allValues.iteritems(), key=operator.itemgetter(1))[0]
            self.usedActionsID.append(self.indexToActionID[nextIndex])
            return self.treeTraversal(nextIndex)
        else: 
            nprint("tree traversal terminated on node %s"%(index))
            availableActions = copy.deepcopy(self.actions)
            for i in self.usedActionsID: 
                availableActions.pop(i, None)
            return (index,availableActions)
        
    def initialiseExplorationNode(self,index,availableActions):
        nprint("expanding %s"%(index))
        for (actionId, (span,numSpan,_)) in availableActions.iteritems() : #initialiseRegions(self.model,self.image,list(set(self.spans[index].keys() + self.uselessPixels))): 
            self.indexToNow += 1
            self.indexToActionID[self.indexToNow] = actionId
            self.initialiseLeafNode(self.indexToNow,index,span,numSpan)
            self.children[index].append(self.indexToNow)
        self.fullyExpanded[index] = True
        self.usedActionsID = []
        return self.children[index]

    def backPropagation(self,index,value): 
        self.cost[index] += value
        self.numberOfVisited[index] += 1
        if self.parent[index] in self.parent : 
            nprint("start backPropagating the value %s from node %s, whose parent node is %s"%(value,index,self.parent[index]))
            self.backPropagation(self.parent[index],value)
        else: 
            nprint("backPropagating ends on node %s"%(index))
            
    # start random sampling and return the eclidean value as the value
    def sampling(self,index,availableActions):
        nprint("start sampling node %s"%(index))
        availableActions2 = copy.deepcopy(availableActions)
        availableActions2.pop(self.indexToActionID[index], None)
        sampleValues = []
        i = 0
        for i in range(MCTS_multi_samples): 
            #allChildren = copy.deepcopy(self.actions)
            (childTerminated, val) = self.sampleNext(self.spans[index],self.numSpans[index],0,availableActions2.keys(),[])
            sampleValues.append(val)
            if childTerminated == True: break
            i += 1
        return (childTerminated, max(sampleValues))
        #return self.sampleNext(self.spans[index],self.numSpans[index])
        #allChildren = initialiseRegions(model,self.image,self.spans[index].keys()) 
    
    def sampleNext(self,spansPath,numSpansPath,depth,availableActionIDs,usedActionIDs): 
        #print spansPath.keys()
        activations1 = applyManipulation(self.activations,spansPath,numSpansPath)
        (newClass,newConfident) = self.predictWithActivations(activations1)
        #print euclideanDistance(self.activations,activations1), newConfident, newClass
        (distMethod,distVal) = controlledSearch
        if distMethod == "euclidean": 
            dist = 1 - euclideanDistance(activations1,self.activations) 
            termValue = 0.0
            termByDist = dist < 1 - distVal
        elif distMethod == "L1": 
            dist = 1 - l1Distance(activations1,self.activations) 
            termValue = 0.0
            termByDist = dist < 1 - distVal
        elif distMethod == "Percentage": 
            dist = 1 - diffPercent(activations1,self.activations)
            termValue = 0.0
            termByDist = dist < 1 - distVal
        elif distMethod == "NumDiffs": 
            dist = self.activations.size - diffPercent(activations1,self.activations) * self.activations.size
            termValue = 0.0
            termByDist = dist < self.activations.size - distVal

        if newClass != self.originalClass: 
            nprint("sampling a path ends in a terminal node with depth %s... "%depth)
            self.decisionTree.addOnePath(dist,spansPath,numSpansPath)
            self.re_training.addDatum(activations1,self.originalClass)
            if self.bestCase[0] < dist: self.bestCase = (dist,spansPath,numSpansPath)
            return (depth == 0, dist)
        elif termByDist == True: 
            nprint("sampling a path ends by controlled search with depth %s ... "%depth)
            return (depth == 0, termValue)
        elif list(set(availableActionIDs)-set(usedActionIDs)) == []: 
            nprint("sampling a path ends with depth %s because no more actions can be taken ... "%depth)
            return (depth == 0, termValue)        
        else: 
            #print("continue sampling node ... ")
            #allChildren = initialiseRegions(self.model,self.activations,spansPath.keys())

            randomActionIndex = random.choice(list(set(availableActionIDs)-set(usedActionIDs))) #random.randint(0, len(allChildren)-1)
            (span,numSpan,_) = self.actions[randomActionIndex]
            availableActionIDs.remove(randomActionIndex)
            usedActionIDs.append(randomActionIndex)
            #print span.keys()
            newSpanPath = self.mergeSpan(spansPath,span)
            newNumSpanPath = self.mergeNumSpan(numSpansPath,numSpan)
            return self.sampleNext(newSpanPath,newNumSpanPath,depth+1,availableActionIDs,usedActionIDs)
            
    def terminalNode(self,index): 
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        (newClass,_) = self.predictWithActivations(activations1)
        return newClass != self.originalClass 
        
    def terminatedByControlledSearch(self,index): 
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        (distMethod,distVal) = controlledSearch
        if distMethod == "euclidean": 
            dist = euclideanDistance(activations1,self.activations) 
        elif distMethod == "L1": 
            dist = l1Distance(activations1,self.activations) 
        elif distMethod == "Percentage": 
            dist = diffPercent(activations1,self.activations)
        elif distMethod == "NumDiffs": 
            dist = diffPercent(activations1,self.activations)
        nprint("terminated by controlled search")
        return dist > distVal 
        
    def applyManipulationToGetImage(self,spans,numSpans):
        activations1 = applyManipulation(self.activations,spans,numSpans)
        if self.layer > -1: 
            return np.squeeze(self.autoencoder.predict(np.expand_dims(activations1,axis=0)))
        else: 
            return activations1
        
    def euclideanDist(self,index): 
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        return euclideanDistance(self.activations,activations1)
        
    def l1Dist(self,index): 
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        return l1Distance(self.activations,activations1)
        
    def diffImage(self,index): 
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        return diffImage(self.activations,activations1)
        
    def diffPercent(self,index): 
        activations1 = applyManipulation(self.activations,self.spans[index],self.numSpans[index])
        return diffPercent(self.activations,activations1)

    def mergeSpan(self,spansPath,span): 
        return mergeTwoDicts(spansPath, span)
        
    def mergeNumSpan(self,numSpansPath,numSpan):
        return mergeTwoDicts(numSpansPath, numSpan)
        
    def showDecisionTree(self):
        self.decisionTree.show()
    
        
#!/usr/bin/env python

"""
main file

author: Xiaowei Huang
"""

import sys
sys.path.append('networks')
sys.path.append('safety_check')
sys.path.append('configuration')
sys.path.append('visualization')
sys.path.append('operation')
sys.path.append('FGSM')


import time
import numpy as np
import copy 
import random
import matplotlib.pyplot as plt
import matplotlib as mpl


from loadData import loadData 
from regionSynth import regionSynth, initialiseRegion
from precisionSynth import precisionSynth
from safety_analysis import safety_analysis
from visualization import visualization

from configuration import *
from basics import *
from networkBasics import *

from searchTree import searchTree
from searchExhaustive import searchExhaustive
from searchMCTS import searchMCTS
from searchAstar import searchAstar

from dataCollection import dataCollection

from operation import cuttingModel

from mnist_network import dynamic_build_model 

from fgsm_loadData import fgsm_loadData
from attacks_th import fgsm
from utils_th import batch_eval
from fgsm import fgsm_main

from inputManipulation import applyManipulation,assignManipulationSimple

import theano
import theano.tensor as T
        
def main():

    model = loadData()
    dc = dataCollection()

    # FGSM
    if test_fgsm == True: 
        for eps in eps_fgsm: 
            fgsm_main(model,eps)
        return
        
    # handle a set of inputs starting from an index
    if dataProcessing == "batch": 
        succNum = 0
        for whichIndex in range(startIndexOfImage,startIndexOfImage + dataProcessingBatchNum):
            print "\n\nprocessing input of index %s in the dataset: " %(str(whichIndex))
            if task == "safety_check": 
                succ = handleOne(model,dc,whichIndex)
                if succ == True: succNum += 1
        dc.addSuccPercent(succNum/float(dataProcessingBatchNum))
        
    # handle a sinextNumSpane input
    else: 
        print "\n\nprocessing input of index %s in the dataset: " %(str(startIndexOfImage))
        if task == "safety_check": 
            handleOne(model,dc,startIndexOfImage)
    if dataProcessing == "batch": 
        dc.provideDetails()
        dc.summarise()
    dc.close()
      
###########################################################################
#
# safety checking
# starting from the first hidden layer
#
############################################################################

def handleOne(model,dc,startIndexOfImage):


    # get an image to interpolate
    global np
    image = NN.getImage(model,startIndexOfImage)
    print("the shape of the input is "+ str(image.shape))
        
    #image = np.array([3.58747339,1.11101673])
    
    dc.initialiseIndex(startIndexOfImage)

    if checkingMode == "stepwise":
        k = startLayer
    elif checkingMode == "specificLayer":
        k = maxLayer
        
    originalModel = copy.deepcopy(model)
    originalImage = copy.deepcopy(image)
    if startLayer > 0: 
        model, image = cuttingModel(model,startLayer,image)
        k = k - startLayer
        kmaxLayer = maxLayer - startLayer
    else: 
        kmaxLayer = maxLayer
        
    while k <= kmaxLayer: 
    
        layerType = getLayerType(model, k)
        start_time = time.time()
            
        # only these layers need to be checked
        if layerType in ["Convolution2D", "Dense"] and searchApproach == "heuristic": 
                    
            dc.initialiseLayer(k)
    
            st = searchTree(image,k)
            st.defineConsideringRegion([(5, 14), (5, 15), (5, 16), (5, 17), (13, 13), (13, 14), (13, 15), (13, 16), (24, 10), (24, 11), (24, 12), (24, 13), (25, 6), (25, 7), (25, 8), (25, 9)])      
            st.addImages(model,[image])

            print "\nstart checking the safety of layer "+str(k)
        
            (originalClass,originalConfident) = NN.predictWithImage(model,image)
            origClassStr = dataBasics.LABELS(int(originalClass))
     
            path0="%s/%s_original_as_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,origClassStr,originalConfident)
            dataBasics.save(-1,originalImage, path0)
            
            # for every layer
            f = 0 
            while f < numOfFeatures : 

                f += 1
                print("\n================================================================")
                print("Round %s of layer %s for image %s"%(f,k,startIndexOfImage))
                index = st.getOneUnexplored()
                imageIndex = copy.deepcopy(index)
                
                #path0="%s/%s_%s.png"%(directory_pic_string,startIndexOfImage,nsn)
                #dataBasics.save(-1,st.images[index], path0)
                        
                # for every image
                # start from the first hidden layer
                t = 0
                while True and index != (-1,-1): 

                    # pick the first element of the queue
                    print "(1) get a manipulated input ..."
                    (image0,span,numSpan,numDimsToMani,_) = st.getInfo(index)
                    
                    print "current layer: %s."%(t)
                    print "current index: %s."%(str(index))
                    
                    path2 = directory_pic_string+"/temp.png"
                    print "current operated image is saved into %s"%(path2)
                    dataBasics.save(index[0],image0,path2)

                    print "(2) synthesise region from %s..."%(span.keys())
                     # ne: next region, i.e., e_{k+1}
                    (nextSpan,nextNumSpan,numDimsToMani) = regionSynth(model,dataset,image0,st.manipulated[t],t,span,numSpan,numDimsToMani)
                    st.addManipulated(t,nextSpan.keys())

                    #print "3) synthesise precision ..."
                    #if not found == True: nextNumSpan = dict(map(lambda (k,v): (k, abs(v-1)), nextNumSpan.iteritems()))
                    # npre : next precision, i.e., p_{k+1}
                    #npre = precisionSynth(model,dataset,image0,t,span,numSpan,nextSpan,nextNumSpan,cp)
                    (nextSpan,nextNumSpan,npre) = precisionSynth(t,nextSpan,nextNumSpan)
                    #print "the precision is %s."%(npre)
                    
                    print "dimensions to be considered: %s"%(nextSpan)
                    #print "dimensions that have been considered before: %s"%(st.manipulated[t])
                    print "spans for the dimensions: %s"%(nextNumSpan)
                
                    if t == k: 
                    
                        # only after reaching the k layer, it is counted as a pass                     
                        print "(3) safety analysis ..."
                        # wk for the set of counterexamples
                        # rk for the set of images that need to be considered in the next precision
                        # rs remembers how many input images have been processed in the last round
                        # nextSpan and nextNumSpan are revised by considering the precision npre
                        (nextSpan,nextNumSpan,rs,wk,rk) = safety_analysis(model,dataset,t,startIndexOfImage,st,index,nextSpan,nextNumSpan,npre)
                        if len(rk) > 0: 
                            rk = (zip (*rk))[0]

                            print "(4) add new images ..."
                            random.seed(time.time())
                            if len(rk) > numOfPointsAfterEachFeature: 
                                rk = random.sample(rk, numOfPointsAfterEachFeature)
                            diffs = diffImage(image0,rk[0])
                            print("the dimensions of the images that are changed in the previous round: %s"%diffs)
                            if len(diffs) == 0: 
                                st.clearManipulated(k)
                                return 
                        
                            st.addImages(model,rk)
                            st.removeProcessed(imageIndex)
                            (re,percent,eudist,l1dist) = reportInfo(image,wk)
                            print "euclidean distance %s"%(euclideanDistance(image,rk[0]))
                            print "L1 distance %s"%(l1Distance(image,rk[0]))
                            print "manipulated percentage distance %s\n"%(diffPercent(image,rk[0]))
                            break
                        else: 
                            st.removeProcessed(imageIndex)
                            break
                    else: 
                        print "(3) add new intermediate node ..."
                        index = st.addIntermediateNode(image0,nextSpan,nextNumSpan,npre,numDimsToMani,index)
                        re = False
                        t += 1
                if re == True: 
                    dc.addManipulationPercentage(percent)
                    print "euclidean distance %s"%(eudist)
                    print "L1 distance %s"%(l1dist)
                    print "manipulated percentage distance %s\n"%(percent)
                    dc.addEuclideanDistance(eudist)
                    dc.addl1Distance(l1dist)
                    (ocl,ocf) = NN.predictWithImage(model,wk[0])
                    dc.addConfidence(ocf)
                    break
                
            if f == numOfFeatures: 
                print "(6) no adversarial example is found in this layer within the distance restriction." 
            st.destructor()
            
        elif layerType in ["Convolution2D", "Dense"] and searchApproach == "exhaustive": 
    
            dc.initialiseLayer(k)
    
            st = searchExhaustive(image,k)
            st.addImages(model,(-1,-1),[(image,NN.predictWithImage(model,image)[1])],[],0)
            print "\nstart checking the safety of layer "+str(k)
            
        
            (originalClass,originalConfident) = NN.predictWithImage(model,image)
            origClassStr = dataBasics.LABELS(int(originalClass))
     
            path0="%s/%s_original_as_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,origClassStr,originalConfident)
            dataBasics.save(-1,originalImage, path0)

            # for every layer
            f = 0 
            #while f <= numOfFeatures : 
            nsn = 0
            while st.emptyQueue() == False and nsn < maxSearchNum :  

                f += 1
                print("\n================================================================")
                print("Round %s of layer %s for image %s"%(f,k,startIndexOfImage))
                index = st.getOneUnexplored()
                st.addVisitedImage(st.images[index])
                
                #path0="%s/%s_%s.png"%(directory_pic_string,startIndexOfImage,nsn)
                #dataBasics.save(-1,st.images[index], path0)
                        
                # for every image
                # start from the first hidden layer
                t = 0
                while True: 

                    # pick the first element of the queue
                    print "(1) get a manipulated input ..."
                    (image0,span,numSpan,numDimsToMani,stepsUpToNow) = st.getInfo(index)

                    print "current layer: %s."%(t)
                    print "current index: %s."%(str(index))
                    print "the number of steps: %s."%(str(stepsUpToNow))
                    print "the number of manipulated that have been modified: %s."%(len(st.manipulated[index]))

                    
                    #path2 = directory_pic_string+"/temp.png"
                    #print "current operated image is saved into %s"%(path2)
                    #dataBasics.save(index[0],image0,path2)

                    print "(2) synthesise region ..."
                     # ne: next region, i.e., e_{k+1}
                    (nextSpan,nextNumSpan,numDimsToMani) = regionSynth(model,dataset,image0,st.manipulated[index][t],t,span,numSpan,numDimsToMani)
                    st.addManipulated(index,t,nextSpan.keys())
                    
                    #print span.keys()
                    #print nextSpan.keys()

                    #print "3) synthesise precision ..."
                    #if not found == True: nextNumSpan = dict(map(lambda (k,v): (k, abs(v-1)), nextNumSpan.iteritems()))
                    # npre : next precision, i.e., p_{k+1}
                    #npre = precisionSynth(model,dataset,image0,t,span,numSpan,nextSpan,nextNumSpan,cp)
                    (nextSpan,nextNumSpan,npre) = precisionSynth(t,nextSpan,nextNumSpan)
                    #print "the precision is %s."%(npre)
                    
                    print "dimensions to be considered: %s"%(nextSpan)
                    #print "dimensions that have been considered before: %s"%(st.manipulated[t])
                    print "spans for the dimensions: %s"%(nextNumSpan)
                
                    if t == k: 
                    
                        # only after reaching the k layer, it is counted as a pass 
                        nsn += 1
                    
                        print "(3) safety analysis ..."
                        # wk for the set of counterexamples
                        # rk for the set of images that need to be considered in the next precision
                        # rs remembers how many input images have been processed in the last round
                        # nextSpan and nextNumSpan are revised by considering the precision npre
                        (nextSpan,nextNumSpan,rs,wk,rk) = safety_analysis(model,dataset,t,startIndexOfImage,st,index,nextSpan,nextNumSpan,npre)

                        print "(4) add new images ..."
                        rk = sorted(rk, key=lambda x: x[1])
                        if len(rk) >= 2 ** featureDims:
                            rk = rk[0:2 ** featureDims-1]
                        
                        # remove identical images
                        rk2 = []
                        if rk != []: 
                            fst = rk[0]
                            remain = rk[1:]
                            while remain != []: 
                                flag = True
                                for i in range(len(remain)):
                                    if np.array_equal(fst[0],remain[i][0]) == True : 
                                        flag = False
                                        break
                                if flag == True: rk2.append(fst)
                                fst = remain[0]
                                remain = remain[1:]
                            rk2.append(fst)
                            print "%s candidate images, but only %s of them are identical."%(len(rk),len(rk2))
                            
                        # remove images that are too far away from the original image
                        rk3 = []
                        for fst in rk2:
                            (distMethod,distVal) = controlledSearch
                            if distMethod == "euclidean": 
                                termByDist = euclideanDistance(fst[0],image) > distVal
                            elif distMethod == "L1": 
                                termByDist = l1Distance(fst[0],image) > distVal
                            termByDist = st.hasVisited(fst[0]) or termByDist
                            if termByDist == False: rk3.append(fst)
                        print "%s identical images, but only %s of them satisfy the distance restriction %s and haven't been visited."%(len(rk2),len(rk3),controlledSearch)
                        
                        
                        rk = rk3    
                        # add cost to the distance for A* algorithm
                        if rk != []: 
                            if costForDijkstra[0] == "euclidean": 
                                scale = (1 - max(zip (*rk)[1]))
                                rk = [(i, c + euclideanDistance(image,i) * scale) for (i,c) in rk]
                            elif costForDijkstra[0] == "l1": 
                                scale = (1 - min(zip (*rk)[1]))
                                #for (i,c) in rk : print("%s --- %s --- %s ---- %s "%(l1Distance(image,i), c, scale, l1Distance(image,i) / scale))
                                rk = [(i, c + l1Distance(image,i) * scale) for (i,c) in rk]
                                                    
                        #diffs = diffImage(image0,rk[0])
                        #print("the dimensions of the images that are changed in the previous round: %s"%diffs)
                        #if len(diffs) == 0: st.clearManipulated(k)
                        
                        if t == 0 :
                            parentIndex = index 
                        else: 
                            parentIndex = st.rootIndexForIntermediateNode(index,t)

                        parentIndex = st.addImages(model,parentIndex,rk,st.manipulated[parentIndex][-1]+st.manipulated[parentIndex][0],stepsUpToNow+1)
                        print "now the queue has %s images (maximum is %s)"%(st.size(),maxQueueSize)
                        print "the number of visited images is %s."%(len(st.visitedImages))

                        print "removing root node %s of %s"%(str(parentIndex),str(index))
                        #newimage = rk[0]
                        st.removeProcessed(parentIndex,t)
                        (re,percent,eudist,l1dist) = reportInfo(image,wk)
                        break
                    else: 
                        print "(3) add new intermediate node ..."
                        index = st.addIntermediateNode(image0,nextSpan,nextNumSpan,npre,numDimsToMani,index)
                        re = False
                        t += 1
                if re == True: 
                    dc.addManipulationPercentage(percent)
                    print "euclidean distance %s"%(eudist)
                    print "L1 distance %s\n"%(l1dist)
                    print "manipulated percentage distance %s\n"%(percent)
                    dc.addEuclideanDistance(eudist)
                    dc.addl1Distance(l1dist)
                    (ocl,ocf) = NN.predictWithImage(model,wk[0])
                    dc.addConfidence(ocf)
                    break
                
            if st.emptyQueue() == True: 
                print "(6) no adversarial example is found in this layer within the distance restriction." 
            st.destructor()


        elif layerType in ["Input"] and searchApproach in ["mcts"]: 
    
            print "directly handling the image ... "
    
            dc.initialiseLayer(k)
            
            (originalClass,originalConfident) = NN.predictWithImage(model,image)
            origClassStr = dataBasics.LABELS(int(originalClass))
            path0="%s/%s_original_as_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,origClassStr,originalConfident)
            dataBasics.save(-1,originalImage, path0)
    
            # initialise a search tree
            st = searchMCTS(model,image,k)
            st.initialiseActions()

            start_time_all = time.time()
            runningTime_all = 0
            numberOfMoves = 0
            while st.terminalNode(st.rootIndex) == False and st.terminatedByControlledSearch(st.rootIndex) == False and runningTime_all <= MCTS_all_maximal_time: 
                print("the number of moves we have made up to now: %s"%(numberOfMoves))
                eudist = st.euclideanDist(st.rootIndex)
                l1dist = st.l1Dist(st.rootIndex)
                percent = st.diffPercent(st.rootIndex)
                diffs = st.diffImage(st.rootIndex)
                print "euclidean distance %s"%(eudist)
                print "L1 distance %s"%(l1dist)
                print "manipulated percentage distance %s"%(percent)
                print "manipulated dimensions %s"%(diffs)

                start_time_level = time.time()
                runningTime_level = 0
                childTerminated = False
                while st.numberOfVisited[st.rootIndex] < maxSearchNum and runningTime_level <= MCTS_level_maximal_time: 
                    (leafNode,availableActions) = st.treeTraversal(st.rootIndex)
                    newNodes = st.initialiseExplorationNode(leafNode,availableActions)
                    for node in newNodes: 
                        (childTerminated, value) = st.sampling(node,availableActions)
                        if childTerminated == True: break
                        st.backPropagation(node,value)
                    if childTerminated == True: break
                    runningTime_level = time.time() - start_time_level   
                    print("best possible one is %s"%(str(st.bestCase)))
                bestChild = st.bestChild(st.rootIndex)
                #st.collectUselessPixels(st.rootIndex)
                st.makeOneMove(bestChild)
                
                image1 = applyManipulation(st.image,st.spans[st.rootIndex],st.numSpans[st.rootIndex])
                diffs = st.diffImage(st.rootIndex)
                path0="%s/%s_temp_%s.png"%(directory_pic_string,startIndexOfImage,len(diffs))
                dataBasics.save(-1,image1,path0)
                (newClass,newConfident) = NN.predictWithImage(model,image1)
                print "confidence: %s"%(newConfident)
                
                if childTerminated == True: break
                
                # store the current best
                (_,bestSpans,bestNumSpans) = st.bestCase
                image1 = applyManipulation(st.image,bestSpans,bestNumSpans)
                path0="%s/%s_currentBest.png"%(directory_pic_string,startIndexOfImage)
                dataBasics.save(-1,image1,path0)
                
                numberOfMoves += 1

            (_,bestSpans,bestNumSpans) = st.bestCase
            #image1 = applyManipulation(st.image,st.spans[st.rootIndex],st.numSpans[st.rootIndex])
            image1 = applyManipulation(st.image,bestSpans,bestNumSpans)
            (newClass,newConfident) = NN.predictWithImage(model,image1)
            newClassStr = dataBasics.LABELS(int(newClass))
            re = newClass != originalClass
            path0="%s/%s_%s_modified_into_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,origClassStr,newClassStr,newConfident)
            dataBasics.save(-1,image1,path0)
            #print np.max(image1), np.min(image1)
            print diffImage(image,image1)
            #plt.imshow(image1 * 255, cmap=mpl.cm.Greys)
            #plt.show()
                
            if re == True: 
                eudist = euclideanDistance(st.image,image1)
                l1dist = l1Distance(st.image,image1)
                percent = diffPercent(st.image,image1)
                print "euclidean distance %s"%(eudist)
                print "L1 distance %s"%(l1dist)
                print "manipulated percentage distance %s"%(percent)
                print "class is changed into %s with confidence %s\n"%(newClassStr, newConfident)
                dc.addEuclideanDistance(eudist)
                dc.addl1Distance(l1dist)
                dc.addManipulationPercentage(percent)
                
            st.destructor()


        elif layerType in ["Input"] and searchApproach in ["heuristic", "exhaustive"]: 
        
            print "directly handling the image ... "
    
            dc.initialiseLayer(k)
    
            # initialise a search tree
            if searchApproach == "heuristic": 
                st = searchTree(image,k)
            # initialise a search queue
            elif searchApproach == "exhaustive":
                st = searchExhaustive(image,k)

            (originalClass,originalConfident) = NN.predictWithImage(model,image)
            st.addImages(model,(-1,-1),[(image,originalConfident)],[],0) 
            
            nsn = 0 
            while st.emptyQueue() == False and nsn < maxSearchNum :  
                nsn += 1
                index = st.getOneUnexplored()
                print "\n==============================================================="
                print "Round: %s"%(str(nsn))
                print "handling image labelled with %s"%(str(index))
                print "number of images remaining in the queue: %s"%(len(st.rk))
                print "number of visited images: %s"%(len(st.visitedImages))

                
                st.addVisitedImage(st.images[index])
                (image0,span,numSpan,numDimsToMani,stepsUpToNow) = st.getInfo(index)
                print "number of steps up to now: %s"%(stepsUpToNow+1)
                print "number of dimensions have been modified: %s"%(len(st.manipulated[index][-1]+st.manipulated[index][0]))



                image1 = applyManipulation(image0,span,numSpan)
                print "appying manipulations on dimensions %s"%(span.keys())
                
                (class1,confident1) = NN.predictWithImage(model,image1)
                re = class1 != originalClass
                print "confidence : %s"%(confident1)
                print "number of dimensions have been modified : %s"%(np.sum(image1 != st.images[(-1,-1)]))
                print "Euclidean distance : %s"%(euclideanDistance(image1,st.images[(-1,-1)]))
                print "L1 distance : %s"%(l1Distance(image1,st.images[(-1,-1)]))
                print "manipulation percentage : %s"%(diffPercent(image1,st.images[(-1,-1)]))

                
                path1 = "%s/%s_%s_%s_into_%s_with_confidence_%s.png"%(directory_pic_string,nsn,startIndexOfImage,originalClass,class1,confident1)
                dataBasics.save(index[0],image1, path1)
                
                if re == True: 
                    path1 = "%s/%s_%s_modified_into_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,originalClass,class1,confident1)
                    dataBasics.save(index[0],image0, path1)
                    wk = [image0]
                    (re,percent,eudist,l1dist) = reportInfo(image,wk)
                
                (distMethod,distVal) = controlledSearch
                if distMethod == "euclidean": 
                    termByDist = euclideanDistance(image1,st.images[(-1,-1)]) > distVal
                elif distMethod == "L1": 
                    termByDist = l1Distance(image1,st.images[(-1,-1)]) > distVal
                elif distMethod == "Percentage": 
                    termByDist = diffPercent(image1,st.images[(-1,-1)]) > distVal
                elif distMethod == "NumDiffs": 
                    termByDist = numDiffs(image1,st.images[(-1,-1)]) + featureDims > distVal
                termByDist = st.hasVisited(image1) or termByDist
                
                #print st.manipulated[index]
                if termByDist == False: 
                    c = st.computeCost(model,image1)
                    print "add images ... "
                    index = st.addImages(model,index,[(image1,c)],st.manipulated[index][-1]+st.manipulated[index][0],stepsUpToNow+1)
                    
                st.removeProcessed(index,0)
                print "removed the image %s"%(str(index))
                
            if re == True: 
                print "euclidean distance %s"%(eudist)
                print "L1 distance %s\n"%(l1dist)
                print "manipulated percentage distance %s\n"%(percent)
                dc.addEuclideanDistance(eudist)
                dc.addl1Distance(l1dist)
                (ocl,ocf) = NN.predictWithImage(model,wk[0])
                dc.addConfidence(ocf)
                break
                
            st.destructor()
 
        elif layerType in ["Input"] and searchApproach in ["Astar"]: 
        
            print "directly handling the image ... "
    
            dc.initialiseLayer(k)
    
            # initialise a search tree
            st = searchAstar(image,k)
            #rg = [(4, 10), (4, 11), (4, 12), (4, 13), (14, 12), (14, 13), (14, 14), (14, 15), (24, 10), (24, 11), (24, 12), (24, 13)]
            rg = [(2, 44, 19), (2, 8, 37), (1, 15, 47), (2, 27, 8), (2, 15, 9), (0, 0, 8), (2, 14, 2), (1, 2, 47), (1, 27, 21), (0, 2, 1), (0, 8, 18), (1, 30, 47), (1, 25, 2), (0, 3, 18), (0, 10, 39), (0, 15, 43), (0, 8, 43), (0, 26, 26), (1, 20, 14), (0, 34, 37), (0, 22, 2), (2, 18, 20), (0, 15, 40), (0, 36, 46), (2, 31, 3), (2, 6, 38), (1, 36, 1), (2, 11, 34), (0, 17, 5), (0, 2, 25), (1, 13, 2), (0, 45, 40), (0, 24, 21), (2, 34, 29), (1, 44, 43), (1, 24, 23), (1, 39, 5), (0, 19, 35), (2, 32, 2), (0, 36, 13), (0, 16, 10), (2, 30, 4), (2, 11, 3), (2, 24, 0), (2, 19, 20), (1, 15, 21), (2, 43, 0), (2, 42, 24), (0, 42, 19), (2, 6, 46), (2, 9, 6), (2, 8, 18), (0, 29, 8), (2, 16, 5), (0, 1, 39), (2, 12, 7), (2, 46, 5), (0, 14, 2), (0, 26, 5), (0, 4, 26), (0, 21, 3), (1, 42, 18), (1, 6, 37), (0, 0, 7), (0, 15, 17), (1, 34, 38), (2, 25, 16), (2, 24, 8), (2, 27, 32), (1, 27, 4), (2, 19, 12), (2, 21, 36), (1, 29, 0), (2, 28, 29), (1, 9, 32), (2, 22, 1), (1, 47, 24), (2, 10, 35), (2, 15, 37), (2, 29, 14), (0, 25, 3), (2, 4, 35), (0, 6, 13), (0, 35, 12), (1, 31, 46), (0, 6, 38), (1, 1, 43), (0, 5, 37), (1, 36, 14), (2, 22, 32), (0, 2, 8), (1, 36, 42), (1, 33, 15), (1, 29, 8), (1, 7, 9), (2, 32, 39), (2, 22, 9), (0, 30, 0), (2, 0, 14), (1, 32, 28), (2, 28, 41), (0, 13, 42), (1, 44, 3), (0, 1, 1), (0, 22, 5), (1, 39, 45), (0, 27, 6), (2, 5, 8), (0, 0, 23), (1, 15, 20), (1, 28, 34), (2, 42, 15), (0, 42, 2), (1, 37, 18), (2, 25, 0), (1, 46, 41), (0, 27, 41), (1, 14, 19), (1, 33, 7), (2, 40, 0), (1, 22, 4), (2, 7, 24), (1, 46, 16), (0, 21, 12), (0, 10, 38), (1, 5, 14), (2, 7, 45), (2, 18, 42), (1, 15, 43), (0, 12, 35), (0, 47, 3), (0, 2, 35), (0, 37, 3), (0, 12, 4), (2, 25, 8), (1, 27, 33), (0, 15, 14), (1, 33, 31), (0, 45, 41), (0, 24, 20), (2, 6, 14), (1, 42, 8), (0, 34, 7), (2, 30, 42), (0, 4, 1), (2, 29, 38), (1, 5, 6), (1, 19, 16), (1, 45, 41), (2, 44, 34), (1, 28, 8), (0, 47, 11), (2, 46, 27), (2, 5, 24), (2, 19, 21), (1, 14, 4), (2, 21, 45), (0, 36, 37), (2, 43, 1), (0, 14, 36), (1, 3, 37), (1, 32, 13), (1, 33, 23), (2, 4, 42), (0, 18, 23), (0, 13, 27), (0, 34, 31), (1, 17, 25), (0, 16, 40), (2, 27, 26), (0, 26, 13), (0, 0, 6), (0, 15, 30), (2, 43, 34), (1, 44, 27), (0, 6, 6), (1, 37, 29), (1, 11, 24), (0, 47, 19), (0, 46, 41), (0, 1, 25), (1, 14, 12), (1, 34, 30), (1, 35, 42), (2, 35, 37), (2, 46, 29), (2, 22, 0), (0, 35, 6), (1, 47, 31), (1, 38, 45), (1, 32, 5), (2, 41, 0), (2, 29, 15), (1, 6, 7), (1, 10, 30), (1, 1, 42), (0, 22, 12), (1, 38, 4), (1, 42, 27), (0, 15, 38), (2, 14, 8), (0, 5, 38), (1, 2, 41), (1, 37, 21), (0, 33, 39), (2, 19, 5), (1, 33, 14), (1, 34, 22), (2, 28, 4), (1, 22, 13), (0, 13, 12), (0, 12, 28), (2, 25, 32), (0, 21, 21), (0, 19, 45), (1, 9, 1), (2, 7, 2), (0, 35, 5), (2, 44, 21), (0, 22, 4), (2, 39, 15), (0, 46, 16), (2, 2, 39), (2, 4, 19), (1, 30, 2), (2, 42, 14), (2, 31, 46), (2, 29, 20), (1, 16, 10), (1, 43, 11), (0, 7, 37), (2, 6, 5), (1, 22, 5), (2, 19, 24), (0, 21, 13), (1, 32, 21), (2, 26, 5), (1, 19, 15), (1, 42, 4), (2, 18, 41), (0, 17, 24), (2, 46, 19), (2, 44, 29), (1, 2, 2), (1, 23, 29), (0, 18, 9), (2, 31, 5), (1, 36, 31), (2, 37, 2), (2, 25, 9), (2, 0, 32), (0, 38, 45), (0, 43, 30), (1, 7, 24), (0, 16, 47), (2, 45, 36), (2, 23, 40), (2, 28, 28), (2, 24, 37), (2, 38, 0), (1, 42, 12), (2, 7, 18), (0, 36, 3), (1, 31, 5), (2, 30, 2), (2, 8, 2), (2, 11, 13), (0, 11, 2), (1, 40, 7), (1, 41, 29), (1, 13, 47), (1, 36, 23), (2, 22, 7), (1, 26, 5), (2, 3, 0), (0, 2, 19), (0, 20, 16), (2, 33, 11), (2, 6, 21), (0, 13, 36), (2, 47, 28), (1, 28, 15), (0, 46, 19), (1, 8, 17), (0, 27, 20), (2, 46, 22), (0, 0, 5), (0, 15, 31), (0, 12, 36), (1, 44, 24), (1, 20, 41), (0, 11, 10), (0, 28, 5), (2, 46, 18), (2, 5, 17), (0, 1, 26), (1, 21, 46), (2, 23, 25), (1, 16, 34), (2, 9, 37), (1, 25, 23), (1, 47, 30), (2, 7, 39), (2, 41, 1), (0, 26, 47), (0, 0, 36), (1, 45, 19), (2, 1, 3), (1, 12, 47), (0, 46, 11), (2, 4, 26), (1, 15, 2), (1, 29, 45), (2, 42, 5), (0, 5, 39), (2, 8, 7), (2, 46, 10), (1, 16, 19), (1, 33, 47), (0, 7, 42), (0, 30, 2), (0, 19, 22), (2, 3, 16), (0, 22, 46), (0, 21, 22), (0, 19, 34), (2, 15, 3), (0, 46, 5), (1, 20, 21), (1, 15, 33), (1, 9, 17), (2, 1, 27), (0, 5, 0), (0, 23, 29), (2, 39, 0), (0, 27, 4), (1, 20, 44), (0, 37, 5), (0, 42, 4), (0, 27, 47), (0, 2, 2), (0, 39, 12), (1, 33, 25), (0, 43, 23), (2, 20, 35), (1, 10, 8), (0, 33, 23), (0, 29, 34), (1, 47, 46), (1, 12, 20), (2, 39, 33), (2, 41, 17), (0, 45, 10), (1, 13, 9), (0, 34, 38), (0, 37, 12), (1, 36, 37), (1, 37, 15), (0, 21, 37), (1, 13, 38), (0, 18, 8), (0, 36, 35), (1, 43, 41), (0, 21, 25), (1, 37, 36), (0, 2, 26), (0, 20, 17), (1, 14, 37), (1, 28, 14), (0, 40, 18), (0, 16, 46), (0, 31, 6), (0, 3, 41), (1, 18, 9), (1, 5, 0), (2, 21, 8), (0, 26, 17), (0, 23, 13), (1, 7, 18), (2, 24, 13), (0, 28, 28), (2, 19, 23), (0, 19, 24), (1, 15, 26), (0, 7, 27), (1, 0, 6), (0, 14, 38), (1, 2, 38), (2, 19, 46), (1, 14, 45), (1, 5, 35), (1, 7, 23), (1, 24, 11), (2, 20, 19), (1, 26, 45), (0, 28, 35), (0, 45, 26), (0, 13, 17), (0, 0, 4), (1, 29, 36), (1, 30, 28), (0, 23, 37), (2, 45, 20), (0, 23, 5), (2, 13, 38), (0, 47, 17), (0, 15, 10), (2, 4, 0), (0, 19, 16), (0, 20, 29), (2, 44, 28), (1, 11, 47), (1, 47, 29), (1, 3, 43), (1, 32, 27), (0, 0, 35), (2, 6, 28), (2, 35, 45), (2, 11, 6), (1, 31, 19), (2, 22, 8), (2, 44, 15), (0, 5, 9), (0, 25, 45), (2, 4, 29), (2, 14, 6), (1, 30, 4), (1, 1, 5), (0, 29, 28), (1, 40, 20), (0, 26, 41), (0, 11, 6), (0, 20, 32), (1, 14, 22), (2, 41, 47), (2, 40, 5), (1, 21, 33), (0, 11, 4), (0, 39, 45), (1, 32, 19), (1, 15, 32), (0, 14, 31), (2, 3, 36), (0, 23, 26), (1, 38, 30), (2, 39, 1), (2, 2, 37), (0, 0, 20), (0, 18, 19), (2, 31, 6), (1, 35, 31), (0, 7, 8), (0, 0, 45), (1, 19, 45), (0, 24, 25), (1, 21, 25), (2, 3, 41), (1, 17, 26), (1, 5, 11), (2, 7, 28), (1, 15, 40), (1, 37, 14), (1, 18, 41), (2, 5, 7), (1, 13, 33), (2, 42, 20), (1, 37, 39), (1, 25, 41), (1, 47, 12), (2, 2, 22), (0, 1, 35), (0, 8, 6), (2, 40, 21), (2, 13, 37), (0, 24, 17), (0, 14, 6), (1, 26, 38), (1, 33, 17), (0, 27, 18), (1, 2, 7), (0, 1, 34), (0, 13, 8), (0, 17, 18), (1, 36, 8), (0, 28, 3), (0, 46, 34), (2, 19, 16), (0, 18, 3), (0, 17, 41), (2, 28, 25), (1, 0, 7), (1, 36, 21), (0, 33, 1), (0, 40, 40), (0, 4, 37), (0, 20, 8), (2, 40, 29), (0, 44, 16), (1, 6, 6), (1, 45, 21), (0, 6, 42), (1, 38, 15), (1, 42, 22), (0, 0, 3), (2, 45, 21), (2, 1, 32), (1, 16, 22), (2, 22, 36), (0, 47, 30), (1, 40, 13), (2, 19, 8), (0, 17, 33), (1, 30, 36), (2, 24, 36), (1, 38, 40), (1, 33, 32), (0, 45, 4), (0, 14, 22), (0, 4, 22), (0, 46, 21), (1, 4, 9), (2, 33, 2), (2, 45, 29), (1, 1, 4), (2, 36, 27), (1, 26, 16), (1, 33, 3), (2, 40, 3), (0, 13, 7), (1, 39, 8), (1, 32, 16), (0, 8, 46), (2, 24, 30), (0, 15, 2), (2, 20, 2), (2, 23, 42), (0, 6, 4), (1, 11, 23), (0, 10, 11), (2, 38, 38), (1, 4, 1), (0, 2, 47), (0, 15, 45), (1, 8, 32), (0, 41, 30), (2, 37, 9), (0, 30, 47), (1, 14, 31), (2, 29, 27), (0, 44, 3), (2, 31, 39), (2, 45, 35), (2, 47, 15), (2, 3, 42), (2, 17, 3), (1, 39, 0), (0, 9, 2), (1, 35, 29), (2, 32, 1), (2, 23, 34), (0, 18, 32), (2, 19, 25), (1, 20, 34), (1, 26, 41), (2, 31, 8), (0, 33, 10), (1, 47, 19), (2, 2, 21), (0, 20, 23), (2, 12, 2), (1, 0, 39), (0, 14, 1), (2, 23, 3), (2, 30, 38), (0, 41, 32), (2, 3, 34), (1, 3, 2), (2, 35, 47), (2, 4, 15), (0, 36, 0), (0, 31, 47), (0, 6, 10), (0, 12, 47), (0, 11, 1), (2, 46, 23), (0, 20, 41), (1, 14, 0), (0, 7, 44), (0, 7, 25), (2, 22, 4), (0, 4, 36), (1, 12, 32), (1, 14, 47), (2, 2, 44), (2, 6, 18), (2, 20, 21), (2, 0, 18), (1, 3, 10), (1, 18, 3), (0, 28, 33), (2, 4, 7), (0, 0, 2), (0, 36, 24), (2, 45, 26), (2, 44, 46), (2, 22, 35), (2, 36, 2), (0, 32, 5), (0, 27, 34), (1, 4, 41), (2, 5, 20), (0, 1, 29), (2, 12, 45), (1, 28, 21), (1, 37, 3), (0, 13, 0), (1, 16, 33), (0, 33, 26), (0, 40, 31), (2, 10, 38), (0, 39, 18), (2, 26, 25), (0, 45, 5), (2, 4, 38), (1, 21, 0), (0, 35, 1), (1, 45, 28), (2, 1, 30), (0, 5, 11), (2, 10, 15), (0, 10, 2), (2, 4, 31), (0, 0, 10), (1, 43, 42), (1, 35, 21), (2, 25, 4), (0, 7, 14), (0, 1, 7), (1, 1, 7), (2, 13, 14), (1, 16, 6), (0, 11, 17), (2, 47, 33), (2, 38, 4), (1, 47, 3), (1, 33, 2), (2, 40, 7), (2, 28, 8), (2, 23, 20), (2, 11, 7), (0, 12, 11), (1, 32, 17), (2, 41, 20), (2, 30, 14), (2, 44, 25), (2, 44, 18), (2, 8, 35), (2, 13, 41), (0, 28, 17), (0, 46, 28), (0, 2, 46), (2, 40, 4), (1, 30, 14), (1, 23, 5), (1, 36, 3), (2, 22, 2), (0, 4, 43), (2, 2, 28), (0, 1, 45), (2, 26, 32), (2, 6, 1), (1, 32, 10), (2, 23, 12), (2, 47, 16), (2, 17, 0), (2, 39, 36), (1, 8, 19), (2, 41, 28), (0, 44, 43), (0, 36, 15), (2, 45, 11), (2, 44, 33), (1, 36, 32), (1, 16, 31), (2, 46, 30), (1, 15, 23), (1, 35, 37), (0, 18, 16), (2, 47, 32), (0, 30, 22), (0, 20, 21), (0, 2, 31), (2, 23, 0), (1, 32, 14), (0, 1, 42), (1, 5, 36), (2, 4, 41), (2, 9, 35), (2, 1, 15), (2, 47, 24), (2, 13, 1), (2, 24, 33), (2, 26, 30), (0, 36, 7), (2, 41, 23), (1, 44, 28), (2, 45, 19), (1, 33, 35), (2, 36, 13), (1, 40, 11), (1, 16, 5), (0, 39, 32), (2, 12, 36), (1, 34, 27), (2, 28, 27), (1, 11, 9), (2, 21, 32), (1, 45, 22), (0, 40, 38), (2, 12, 13), (2, 29, 8), (1, 6, 4), (0, 14, 30), (2, 9, 43), (1, 12, 35), (1, 40, 40), (0, 35, 38), (2, 4, 6), (0, 0, 1), (1, 8, 22), (1, 35, 12), (0, 30, 23), (1, 23, 37), (1, 46, 39), (0, 28, 9), (2, 26, 2), (1, 32, 39), (0, 42, 33), (0, 13, 1), (2, 10, 16), (0, 19, 46), (0, 20, 22), (1, 47, 35), (1, 20, 25), (1, 21, 3), (0, 13, 40), (1, 10, 35), (1, 12, 43), (2, 24, 17), (0, 28, 40), (2, 37, 40), (2, 4, 30), (0, 0, 9), (1, 35, 11), (1, 35, 20), (0, 24, 37), (2, 42, 0), (1, 2, 46), (2, 47, 34), (0, 7, 38), (2, 23, 21), (1, 25, 3), (2, 3, 44), (2, 13, 7), (1, 5, 12), (1, 20, 17), (2, 39, 4), (0, 28, 16), (2, 5, 2), (0, 40, 7), (1, 35, 28), (1, 22, 46), (1, 23, 21), (0, 12, 6), (0, 35, 3), (1, 42, 3), (0, 39, 8), (1, 32, 46), (0, 25, 18), (1, 7, 27), (1, 24, 5), (2, 23, 13), (1, 44, 42), (2, 7, 25), (1, 46, 30), (1, 26, 33), (0, 4, 3), (2, 1, 0), (2, 33, 37), (1, 15, 45), (1, 8, 7), (2, 44, 32), (2, 8, 42), (0, 28, 24), (0, 46, 39), (1, 8, 46), (1, 35, 36), (2, 42, 25), (0, 23, 40), (2, 11, 41), (0, 29, 15), (2, 35, 35), (0, 39, 0), (1, 7, 19), (1, 30, 23), (2, 20, 31), (1, 10, 20), (0, 6, 47), (2, 0, 20), (0, 4, 27), (0, 24, 24), (0, 27, 17), (2, 21, 12), (0, 41, 9), (1, 1, 9), (1, 16, 20), (0, 33, 45), (0, 28, 0), (2, 14, 46), (0, 39, 33), (1, 14, 2), (2, 28, 33), (1, 34, 28), (1, 35, 44), (2, 42, 33), (0, 39, 4), (2, 37, 31), (1, 47, 25), (0, 3, 2), (2, 16, 2), (0, 39, 24), (2, 12, 12), (2, 41, 2), (2, 33, 46), (0, 14, 11), (1, 31, 47), (1, 27, 6), (0, 45, 7), (0, 38, 11), (2, 24, 40), (1, 27, 36), (1, 41, 3), (0, 38, 8), (2, 4, 25), (0, 0, 0), (2, 22, 12), (0, 43, 33), (0, 41, 17), (0, 10, 42), (2, 22, 33), (1, 14, 1), (0, 2, 9), (0, 24, 5), (1, 21, 37), (0, 4, 24), (0, 35, 44), (2, 47, 17), (2, 34, 38), (0, 40, 29), (2, 43, 14), (0, 21, 27), (2, 19, 34), (1, 32, 31), (2, 26, 31), (0, 26, 18), (2, 33, 38), (1, 12, 4), (0, 13, 41), (1, 10, 36)]
            workingImage = st.defineConsideringRegion(rg)   
            
            path1 = "%s/%s_workingImage.png"%(directory_pic_string,startIndexOfImage)
            dataBasics.save(0,workingImage, path1)
            
            eud = euclideanDistance(workingImage,st.images[(-1,-1)])
            print eud

           
            (class1,confident1) = NN.predictWithImage(model,workingImage)
            print dataBasics.LABELS(int(class1))            

            st.initialiseActions()
            

            (originalClass,originalConfident) = NN.predictWithImage(model,image)
            st.addImages(model,(-1,-1),[(workingImage,confident1)],[],0) 
            
            f = open('%s/%s_info.txt'%(directory_pic_string,startIndexOfImage), 'w')
            
            nsn = 0 
            while st.emptyQueue() == False and nsn < maxSearchNum :  
                nsn += 1
                index = st.getOneUnexplored()
                print "\n==============================================================="
                print "Round: %s"%(str(nsn))
                print "handling image labelled with %s"%(str(index))
                print "number of images remaining in the queue: %s"%(len(st.rk))
                print "number of visited images: %s"%(len(st.visitedImages))

                
                st.addVisitedImage(st.images[index])
                (image0,span,numSpan,numDimsToMani,stepsUpToNow) = st.getInfo(index)
                print "number of steps up to now: %s"%(stepsUpToNow+1)
                print "number of dimensions have been modified: %s"%(len(st.manipulated[index][-1]+st.manipulated[index][0]))

                image1 = assignManipulationSimple(image0,span,numSpan)
                print "appying manipulations on dimensions %s, with changes %s"%(span.keys(),span.values())
                st.reportCurrentValue(image1)
                
                (class1,confident1) = NN.predictWithImage(model,image1)
                re = class1 != originalClass
                eud = euclideanDistance(image1,st.images[(-1,-1)])
                l1d = l1Distance(image1,st.images[(-1,-1)])
                print "confidence : %s"%(confident1)
                print "number of dimensions have been modified : %s"%(np.sum(image1 != st.images[(-1,-1)]))
                print "Euclidean distance : %s"%(eud)
                print "L1 distance : %s"%(l1d)
                print "manipulation percentage : %s"%(diffPercent(image1,st.images[(-1,-1)]))

                
                #path1 = "%s/%s_%s_%s_into_%s_with_confidence_%s.png"%(directory_pic_string,nsn,startIndexOfImage,originalClass,class1,confident1)
                #dataBasics.save(index[0],image1, path1)
                
                if re == True: 
                    print "class changed from %s into %s"%(originalClass,class1)
                    path1 = "%s/%s_%s_modified_into_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,originalClass,class1,confident1)
                    f.write(path1)
                    f.write("\nL1 distance : %s\n"%(l1d))
                    f.write("Euclidean distance : %s\n"%(eud))
                    f.write("confidence : %s\n\n"%(confident1))
                    dataBasics.save(index[0],image0, path1)
                    wk = [image0]
                    (re,percent,eudist,l1dist) = reportInfo(image,wk)
                    #break
                
                (distMethod,distVal) = controlledSearch
                if distMethod == "euclidean": 
                    termByDist = euclideanDistance(image1,st.images[(-1,-1)]) > distVal
                elif distMethod == "L1": 
                    termByDist = l1Distance(image1,st.images[(-1,-1)]) > distVal
                elif distMethod == "Percentage": 
                    termByDist = diffPercent(image1,st.images[(-1,-1)]) > distVal
                elif distMethod == "NumDiffs": 
                    termByDist = numDiffs(image1,st.images[(-1,-1)]) + featureDims > distVal
                termByDist = st.hasVisited(image1) or termByDist
                
                #print st.manipulated[index]
                if termByDist == False: 
                    c = st.computeCost(model,image1)
                    print "add images ... "
                    index = st.addImages(model,index,[(image1,c)],st.manipulated[index][-1]+st.manipulated[index][0],stepsUpToNow+1)
                    
                st.removeProcessed(index,0)
                print "removed the image %s"%(str(index))
                
            if re == True: 
                print "euclidean distance %s"%(eudist)
                print "L1 distance %s\n"%(l1dist)
                print "manipulated percentage distance %s\n"%(percent)
                dc.addEuclideanDistance(eudist)
                dc.addl1Distance(l1dist)
                (ocl,ocf) = NN.predictWithImage(model,wk[0])
                dc.addConfidence(ocf)
                f.close()
                break
                
            st.destructor()

                
        runningTime = time.time() - start_time   
        dc.addRunningTime(runningTime)
        if re == True and exitWhen == "foundFirst": 
            break
        k += 1    
     
    print("Please refer to the file %s for statistics."%(dc.fileName))
    if re == True: 
        return True
    else: return False
    

def reportInfo(image,wk):

    # exit only when we find an adversarial example
    if wk == []:    
        print "(5) no adversarial example is found in this round."  
        return (False,0,0,0)
    else: 
        print "(5) an adversarial example has been found."
        image0 = wk[0]
        eudist = euclideanDistance(image,image0)
        l1dist = l1Distance(image,image0)
        percent = diffPercent(image,image0)
        return (True,percent,eudist,l1dist)
        
if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    
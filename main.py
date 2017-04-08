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
            rg = [(4, 10), (4, 11), (4, 12), (4, 13), (14, 12), (14, 13), (14, 14), (14, 15), (24, 10), (24, 11), (24, 12), (24, 13)]
            workingImage = st.defineConsideringRegion(rg)      
            (class1,confident1) = NN.predictWithImage(model,workingImage)
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
    
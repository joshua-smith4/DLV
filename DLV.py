#!/usr/bin/env python

"""
main file

author: Xiaowei Huang
"""

import sys
sys.path.append('networks')
sys.path.append('safety_check')
sys.path.append('configuration')
sys.path.append('basics')


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

from configuration import *
from basics import *
from networkBasics import *

from searchTree import searchTree
from searchMCTS import searchMCTS
from dataCollection import dataCollection

from inputManipulation import applyManipulation,assignManipulationSimple

import theano
import theano.tensor as T
        
def main():

    model = loadData()
    dc = dataCollection()
                    
    # handle a set of inputs starting from an index
    succNum = 0
    for whichIndex in range(startIndexOfImage,startIndexOfImage + dataProcessingBatchNum):
        print "\n\nprocessing input of index %s in the dataset: " %(str(whichIndex))
        succ = handleOne(model,dc,whichIndex)
        if succ == True: succNum += 1
    dc.addSuccPercent(succNum/float(dataProcessingBatchNum))
    dc.provideDetails()
    dc.summarise()
    dc.close()
      
###########################################################################
#
# safety checking
# starting from the a specified hidden layer
#
############################################################################

## how many branches to expand 
numOfPointsAfterEachFeature = 1

def handleOne(model,dc,startIndexOfImage):


    # get an image to interpolate
    global np
    image = NN.getImage(model,startIndexOfImage)
    print("the shape of the input is "+ str(image.shape))
            
    dc.initialiseIndex(startIndexOfImage)
    originalImage = copy.deepcopy(image)

    if checkingMode == "stepwise":
        k = startLayer
    elif checkingMode == "specificLayer":
        k = maxLayer
        
    while k <= maxLayer: 
    
        layerType = getLayerType(model, k)
        start_time = time.time()
            
        # only these layers need to be checked
        if layerType in ["Convolution2D", "Dense"] and searchApproach == "heuristic": 
                    
            dc.initialiseLayer(k)
    
            st = searchTree(image,k)
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

                    (nextSpan,nextNumSpan,npre) = precisionSynth(t,nextSpan,nextNumSpan)
                    
                    print "dimensions to be considered: %s"%(nextSpan)
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
                while runningTime_level <= MCTS_level_maximal_time: 
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
            print("difference between images: %s"%(diffImage(image,image1)))
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


        elif layerType in ["Input"] and searchApproach in ["heuristic"]: 
        
            print "directly handling the image ... "
    
            dc.initialiseLayer(k)
    
            # initialise a search tree
            st = searchTree(image,k)

            (originalClass,originalConfident) = NN.predictWithImage(model,image)
            st.addImages(model,(-1,-1),[(image,originalConfident)],[],0) 
            
            nsn = 0 
            while st.emptyQueue() == False :  
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
    
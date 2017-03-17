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
from dataCollection import dataCollection

from mnist_network import dynamic_build_model 
from operation import operation

from fgsm_loadData import fgsm_loadData
from attacks_th import fgsm
from utils_th import batch_eval
from fgsm import fgsm_main

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
    
    # visualising the image
    #visualization(model,dataset,startIndexOfImage,image)
    #return 

    # operate on the NN model
    #operation(model,3,image)
    
    dc.initialiseIndex(startIndexOfImage)

    if checkingMode == "stepwise":
        k = startLayer
    elif checkingMode == "specificLayer":
        k = maxLayer
        
    while k <= maxLayer: 
    
        layerType = getLayerType(model, k)
        start_time = time.time()
        
        # only these layers need to be checked
        if layerType in ["Convolution2D", "Dense"]: 
    
            dc.initialiseLayer(k)
    
            # initialise a search tree
            if searchApproach == "heuristic": 
                st = searchTree(image,k)
            # initialise a search queue
            elif searchApproach == "exhaustive":
                st = searchExhaustive(image,k)

            st.addImages(model,[(image,NN.predictWithImage(model,image)[1])],0)
            print "\nstart checking the safety of layer "+str(k)
            print "the current context is %s"%(st.numSpans)
        
            (originalClass,originalConfident) = NN.predictWithImage(model,image)
            origClassStr = dataBasics.LABELS(int(originalClass))
     
            path0="%s/%s_original_as_%s_with_confidence_%s.png"%(directory_pic_string,startIndexOfImage,origClassStr,originalConfident)
            dataBasics.save(-1,image, path0)

            # for every layer
            f = 0 
            while f <= numOfFeatures: 

                f += 1
                index = st.getOneUnexplored()
                imageIndex = copy.deepcopy(index)
            
                #howfar = st.getHowFar(index[0],0)
            
                # for every image
                # start from the first hidden layer
                t = 0
                while True: 



                    #print "\nhow many dimensions have been changed: %s."%(len(st.manipulated[-1]))

                    # pick the first element of the queue
                    print "\n(1) get a manipulated input ..."
                    (image0,span,numSpan,numDimsToMani,stepsUpToNow) = st.getInfo(index)
                    
                    print "current index: %s."%(str(index))
                    print "the number of steps: %s."%(str(stepsUpToNow))
                    print "current layer: %s."%(t)
                    
                    path2 = directory_pic_string+"/temp.png"
                    print "current operated image is saved into %s"%(path2)
                    dataBasics.save(index[0],image0,path2)

                    print "(2) synthesise region ..."
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
                        print "(3) safety analysis ..."
                        # wk for the set of counterexamples
                        # rk for the set of images that need to be considered in the next precision
                        # rs remembers how many input images have been processed in the last round
                        # nextSpan and nextNumSpan are revised by considering the precision npre
                        (nextSpan,nextNumSpan,rs,wk,rk) = safety_analysis(model,dataset,t,startIndexOfImage,st,index,nextSpan,nextNumSpan,npre)

                        print "(4) add new images ..."
                        rk = sorted(rk, key=lambda x: x[1])
                        if len(rk) >= numOfPointsAfterEachFeature:
                            rk = rk[0:numOfPointsAfterEachFeature-1]
                        
                        if rk != []: 
                            rk2 = []
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
                            rk = rk2
                            
                        if costForDijkstra == "euclidean": 
                            scale = (1 - max(zip (*rk)[1]))
                            rk = [(i, c + euclideanDistance(image,i) * scale) for (i,c) in rk]
                        elif costForDijkstra == "l1": 
                            scale = (1 - min(zip (*rk)[1]))
                            #for (i,c) in rk : print("%s --- %s --- %s ---- %s "%(l1Distance(image,i), c, scale, l1Distance(image,i) / scale))
                            rk = [(i, c + l1Distance(image,i) * scale) for (i,c) in rk]
                                                    
                        #diffs = diffImage(image0,rk[0])
                        #print("the dimensions of the images that are changed in the previous round: %s"%diffs)
                        #if len(diffs) == 0: st.clearManipulated(k)
                        
                        if searchApproach == "exhaustive": 
                            st.addImages(model,rk,stepsUpToNow+1)
                        elif  searchApproach == "heuristic": 
                            st.addImages(model,(zip(*rk))[0],stepsUpToNow+1)
                            
                        #newimage = rk[0]
                        st.removeProcessed(imageIndex)
                        (re,percent,eudist,l1dist) = reportInfo(image,wk)
                        break
                    else: 
                        print "3) add new intermediate node ..."
                        index = st.addIntermediateNode(image0,nextSpan,nextNumSpan,npre,numDimsToMani,index)
                        re = False
                        t += 1
                if re == True: 
                    dc.addManipulationPercentage(percent)
                    print "euclidean distance %s"%(eudist)
                    print "L1 distance %s\n"%(l1dist)
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
        print "(5) no adversarial example is found in this layer."  
        return (False,0,0,0)
    else: 
        print "(5) an adversarial example has been found."
        image0 = wk[0]
        diffs = diffImage(image,image0)
        eudist = euclideanDistance(image,image0)
        l1dist = l1Distance(image,image0)
        elts = len(diffs)
        if len(image0.shape) == 2: 
            percent = elts / float(len(image0)*len(image0[0]))
        elif len(image0.shape) == 1:
            percent = elts / float(len(image0))
        elif len(image0.shape) == 3:
            percent = elts / float(len(image0)*len(image0[0])*len(image0[0][0]))
        return (True,percent,eudist,l1dist)
        
if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    
#!/usr/bin/env python

"""
Define paramters
author: Xiaowei Huang
"""

def usual_configuration(dataset):

    if dataset == "twoDcurve": 
    
        # which image to start with or work with 
        # from the database
        startIndexOfImage = 0
        
        # the start layer to work from 
        startLayer = 0

        # the maximal layer to work until 
        maxLayer = 1
        
        # search approach
        #searchApproach = "heuristic"
        searchApproach = "mcts"

        ## number of features of each layer 
        # in the paper, dims_L = numOfFeatures * featureDims
        numOfFeatures = 0
        
        ## control by distance
        controlledSearch = ("euclidean",0.1)
        #controlledSearch = ("L1",0.05)
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 300
        MCTS_all_maximal_time = 1800
        MCTS_multi_samples = 3

        # point-based or line-based, or only work with a specific point
        #enumerationMethod = "convex"
        enumerationMethod = "line"


        #checkingMode = "specificLayer"
        checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        exitWhen = "foundAll"
        #exitWhen = "foundFirst"
        
        return (startIndexOfImage,startLayer,maxLayer,searchApproach,numOfFeatures,controlledSearch,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,enumerationMethod,checkingMode,exitWhen)
        
    elif dataset == "mnist": 

        # which image to start with or work with 
        # from the database
        startIndexOfImage = 5422
        
        # the start layer to work from 
        startLayer = -1
        # the maximal layer to work until 
        maxLayer = -1
        
        # search approach
        #searchApproach = "heuristic"
        searchApproach = "mcts"

        ## number of features of each layer 
        numOfFeatures = 156 # 921 # 
        
        ## control by distance
        #controlledSearch = ("euclidean",0.3)
        controlledSearch = ("L1",0.02)
        #controlledSearch = ("Percentage",0.12)
        #controlledSearch = ("NumDiffs",30)
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 60
        MCTS_all_maximal_time = 300
        MCTS_multi_samples = 5
        
        # use linear restrictions or conv filter restriction
        inverseFunction = "point"
        #inverseFunction = "area"

        # point-based or line-based, or only work with a specific point
        enumerationMethod = "convex"
        #enumerationMethod = "line"

        #checkingMode = "specificLayer"
        checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        #exitWhen = "foundAll"
        exitWhen = "foundFirst"
        
    
        return (startIndexOfImage,startLayer,maxLayer,searchApproach,numOfFeatures,controlledSearch,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,enumerationMethod,checkingMode,exitWhen)
        
        
    elif dataset == "gtsrb": 

        # which image to start with or work with 
        # from the database
        startIndexOfImage = 4894
        
        # the start layer to work from 
        startLayer = -1
        # the maximal layer to work until 
        maxLayer = -1
        
        # search approach
        #searchApproach = "heuristic"
        searchApproach = "mcts"

        ## number of features of each layer 
        numOfFeatures = 307 # 3000
        
        ## control by distance
        #controlledSearch = ("euclidean",0.3)
        controlledSearch = ("L1",0.15)
        #controlledSearch = ("Percentage",0.12)
        #controlledSearch = ("NumDiffs",30)
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 300
        MCTS_all_maximal_time = 1800
        MCTS_multi_samples = 3

        # point-based or line-based, or only work with a specific point
        enumerationMethod = "convex"
        #enumerationMethod = "line"


        #checkingMode = "specificLayer"
        checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        #exitWhen = "foundAll"
        exitWhen = "foundFirst"
        
    
        return (startIndexOfImage,startLayer,maxLayer,searchApproach,numOfFeatures,controlledSearch,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,enumerationMethod,checkingMode,exitWhen)
        
    elif dataset == "cifar10": 
    
        # which image to start with or work with 
        # from the database
        startIndexOfImage = 385
        
        # the start layer to work from 
        startLayer = -1
        # the maximal layer to work until 
        maxLayer = -1
        
        # search approach
        searchApproach = "heuristic"
        #searchApproach = "mcts"

        ## number of features of each layer 
        numOfFeatures = 204 # 540
        
        ## control by distance
        #controlledSearch = ("euclidean",0.3)
        controlledSearch = ("L1",0.25)
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 300
        MCTS_all_maximal_time = 1800
        MCTS_multi_samples = 3

        # point-based or line-based, or only work with a specific point
        enumerationMethod = "convex"
        #enumerationMethod = "line"


        #checkingMode = "specificLayer"
        checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        #exitWhen = "foundAll"
        exitWhen = "foundFirst"
        
        return (startIndexOfImage,startLayer,maxLayer,searchApproach,numOfFeatures,controlledSearch,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,enumerationMethod,checkingMode,exitWhen)

    elif dataset == "imageNet": 
    
        # which image to start with or work with 
        # from the database
        startIndexOfImage = 1
        
        # the start layer to work from 
        startLayer = 0
        # the maximal layer to work until 
        maxLayer = 1

        # search approach
        #searchApproach = "heuristic"
        searchApproach = "mcts"

        ## number of features of each layer 
        numOfFeatures = 20000
        
        ## control by distance
        controlledSearch = ("euclidean",0.1)
        #controlledSearch = ("L1",0.05)
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 300
        MCTS_all_maximal_time = 1800
        MCTS_multi_samples = 3

        # point-based or line-based, or only work with a specific point
        enumerationMethod = "convex"
        #enumerationMethod = "line"
        #enumerationMethod = "point"

        checkingMode = "specificLayer"
        #checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        #exitWhen = "foundAll"
        exitWhen = "foundFirst"
        
    
        return (startIndexOfImage,startLayer,maxLayer,searchApproach,numOfFeatures,controlledSearch,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,enumerationMethod,checkingMode,exitWhen)
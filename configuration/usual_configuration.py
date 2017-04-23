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
        #searchApproach = "exhaustive"
        searchApproach = "mcts"

        ## number of features of each layer 
        # in the paper, dims_L = numOfFeatures * featureDims
        numOfFeatures = 0
        
        ## control by distance
        controlledSearch = ("euclidean",0.1)
        #controlledSearch = ("L1",0.05)
        
        ## maximal number of searches 
        maxSearchNum = 1000
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 300
        MCTS_all_maximal_time = 1800
        MCTS_multi_samples = 3

        #cost function used to compute the distance from the starting point
        #costForDijkstra = ("euclidean",0.0)
        costForDijkstra = ("l1",0.0)
        
        #maximum number of elements in the search queue
        maxQueueSize = 5
        
        # the rate that we do not use heuristic but the random sampling
        # to get the next feature 
        explorationRate = 0.5

        # use linear restrictions or conv filter restriction
        inverseFunction = "point"
        #inverseFunction = "area"

        # point-based or line-based, or only work with a specific point
        #enumerationMethod = "convex"
        enumerationMethod = "line"

        # do we need to repeatedly select an updated input neuron
        #repeatedManipulation = "allowed"
        repeatedManipulation = "disallowed"

        #checkingMode = "specificLayer"
        checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        exitWhen = "foundAll"
        #exitWhen = "foundFirst"
        
        return (startIndexOfImage,startLayer,maxLayer,searchApproach,numOfFeatures,maxQueueSize,explorationRate,controlledSearch,maxSearchNum,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,inverseFunction,enumerationMethod,repeatedManipulation,checkingMode,exitWhen)
        
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
        #searchApproach = "exhaustive"
        searchApproach = "mcts"
        #searchApproach = "Astar"

        ## number of features of each layer 
        numOfFeatures = 156 # 921 # 
        
        ## control by distance
        #controlledSearch = ("euclidean",0.3)
        controlledSearch = ("L1",0.02)
        #controlledSearch = ("Percentage",0.12)
        #controlledSearch = ("NumDiffs",30)
        
        ## maximal number of searches 
        maxSearchNum = 20000
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 60
        MCTS_all_maximal_time = 300
        MCTS_multi_samples = 5
        
        #cost function used to compute the distance from the starting point
        # (distance, scale). A* is distance + scale * confidence
        #costForDijkstra = ("euclidean",0.0)
        costForDijkstra = ("l1",1.0)
        
        
        #maximum number of elements in the search queue
        maxQueueSize = 5000
        
        # the rate that we do not use heuristic but the random sampling
        # to get the next feature 
        explorationRate = 0.0
        
        # use linear restrictions or conv filter restriction
        inverseFunction = "point"
        #inverseFunction = "area"

        # point-based or line-based, or only work with a specific point
        enumerationMethod = "convex"
        #enumerationMethod = "line"

        # do we need to repeatedly select an updated input neuron
        # disallowed is needed for mcts and exhaustive search
        #repeatedManipulation = "allowed"
        repeatedManipulation = "disallowed"

        #checkingMode = "specificLayer"
        checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        #exitWhen = "foundAll"
        exitWhen = "foundFirst"
        
    
        return (startIndexOfImage,startLayer,maxLayer,searchApproach,numOfFeatures,maxQueueSize,explorationRate,controlledSearch,maxSearchNum,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,inverseFunction,enumerationMethod,repeatedManipulation,checkingMode,exitWhen)
        
        
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
        #searchApproach = "exhaustive"
        searchApproach = "mcts"
        #searchApproach = "Astar"

        ## number of features of each layer 
        numOfFeatures = 307 # 3000
        
        ## control by distance
        #controlledSearch = ("euclidean",0.3)
        controlledSearch = ("L1",0.15)
        #controlledSearch = ("Percentage",0.12)
        #controlledSearch = ("NumDiffs",30)
        
        ## maximal number of searches 
        maxSearchNum = 1000
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 300
        MCTS_all_maximal_time = 1800
        MCTS_multi_samples = 3
        
        #cost function used to compute the distance from the starting point
        #costForDijkstra = ("euclidean",0.0)
        costForDijkstra = ("l1",0.0)

        #maximum number of elements in the search queue
        maxQueueSize = 5
        
        # the rate that we do not use heuristic but the random sampling
        # to get the next feature 
        explorationRate = 0.0

        # use linear restrictions or conv filter restriction
        inverseFunction = "point"
        #inverseFunction = "area"

        # point-based or line-based, or only work with a specific point
        enumerationMethod = "convex"
        #enumerationMethod = "line"

        # do we need to repeatedly select an updated input neuron
        #repeatedManipulation = "allowed"
        repeatedManipulation = "disallowed"

        #checkingMode = "specificLayer"
        checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        #exitWhen = "foundAll"
        exitWhen = "foundFirst"
        
    
        return (startIndexOfImage,startLayer,maxLayer,searchApproach,numOfFeatures,maxQueueSize,explorationRate,controlledSearch,maxSearchNum,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,inverseFunction,enumerationMethod,repeatedManipulation,checkingMode,exitWhen)
        
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
        #searchApproach = "exhaustive"
        #searchApproach = "mcts"

        ## number of features of each layer 
        numOfFeatures = 204 # 540
        
        ## control by distance
        #controlledSearch = ("euclidean",0.3)
        controlledSearch = ("L1",0.25)
        
        ## maximal number of searches 
        maxSearchNum = 1000
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 300
        MCTS_all_maximal_time = 1800
        MCTS_multi_samples = 3
        
        #cost function used to compute the distance from the starting point
        #costForDijkstra = ("euclidean",0.0)
        costForDijkstra = ("l1",0.0)
        
        #maximum number of elements in the search queue
        maxQueueSize = 5
        
        # the rate that we do not use heuristic but the random sampling
        # to get the next feature 
        explorationRate = 0.0

        # use linear restrictions or conv filter restriction
        inverseFunction = "point"
        #inverseFunction = "area"

        # point-based or line-based, or only work with a specific point
        enumerationMethod = "convex"
        #enumerationMethod = "line"

        # do we need to repeatedly select an updated input neuron
        #repeatedManipulation = "allowed"
        repeatedManipulation = "disallowed"

        #checkingMode = "specificLayer"
        checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        #exitWhen = "foundAll"
        exitWhen = "foundFirst"
        
        return (startIndexOfImage,startLayer,maxLayer,searchApproach,numOfFeatures,maxQueueSize,explorationRate,controlledSearch,maxSearchNum,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,inverseFunction,enumerationMethod,repeatedManipulation,checkingMode,exitWhen)

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
        #searchApproach = "exhaustive"
        searchApproach = "mcts"

        ## number of features of each layer 
        numOfFeatures = 20000
        
        ## control by distance
        controlledSearch = ("euclidean",0.1)
        #controlledSearch = ("L1",0.05)
        
        ## maximal number of searches 
        maxSearchNum = 1000
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 300
        MCTS_all_maximal_time = 1800
        MCTS_multi_samples = 3
        
        #cost function used to compute the distance from the starting point
        #costForDijkstra = ("euclidean",0.0)
        costForDijkstra = ("l1",0.0)
        
        #maximum number of elements in the search queue
        maxQueueSize = 5
        
        # the rate that we do not use heuristic but the random sampling
        # to get the next feature 
        explorationRate = 0.5

        # use linear restrictions or conv filter restriction
        inverseFunction = "point"
        #inverseFunction = "area"

        # point-based or line-based, or only work with a specific point
        enumerationMethod = "convex"
        #enumerationMethod = "line"
        #enumerationMethod = "point"

        # do we need to repeatedly select an updated input neuron
        #repeatedManipulation = "allowed"
        repeatedManipulation = "disallowed"

        checkingMode = "specificLayer"
        #checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        #exitWhen = "foundAll"
        exitWhen = "foundFirst"
        
    
        return (startIndexOfImage,startLayer,maxLayer,searchApproach,numOfFeatures,maxQueueSize,explorationRate,controlledSearch,maxSearchNum,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,inverseFunction,enumerationMethod,repeatedManipulation,checkingMode,exitWhen)
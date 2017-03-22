#!/usr/bin/env python

import numpy as np
import math
import ast
import copy
import random
import time
import stopit



from scipy import ndimage



def applyManipulation(image,span,numSpan):

    for elt in span.keys(): 
        if len(elt) == 2: 
            (fst,snd) = elt 
            if 1 - image[fst][snd] < image[fst][snd] : image[fst][snd] -= numSpan[elt] * span[elt]
            else: image[fst][snd] += numSpan[elt] * span[elt]
        elif len(elt) == 3: 
            (fst,snd,thd) = elt 
            if 1 - image[fst][snd][thd] < image[fst][snd][thd] : image[fst][snd][thd] -= numSpan[elt] * span[elt]
            else: image[fst][snd][thd] += numSpan[elt] * span[elt]
        if image[fst][snd] < 0: image[fst][snd] = 0
        elif image[fst][snd] > 1: image[fst][snd] = 1
            
    return image 
    
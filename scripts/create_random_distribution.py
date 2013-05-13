# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:05:40 2013

@author: Jan
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import pyramid.magcreator as mc
import time
import pdb, traceback, sys
from numpy import pi


def phase_from_mag():
    
    count = 10
    dim = (128, 128)    
    
    random.seed(42)
    
    for i in range(count):
        
        x = random.rand
    
    
if __name__ == "__main__":
    try:
        phase_from_mag()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 11:15:38 2013

@author: Jan
"""

import numpy as np
import pyramid.magcreator as mc
import pyramid.dataloader as dl
import pyramid.phasemap as pm
import time
import pdb, traceback, sys
from numpy import pi


def phase_from_mag():
    '''Calculate and display the phase map from a given magnetization.
    Arguments:
        None
    Returns:
        None
    
    '''
    # TODO: Input via GUI
    filename = '../output/output.txt'
    b_0 = 1.0  # in T    
    dim = (3, 3)  # in px (y,x)
    res = 10.0  # in nm
    beta = pi/4    
    
    center = (1, 1)  # in px (y,x) index starts with 0!
    width  = (1, 1)  # in px (y,x)
    mag_shape = mc.slab(dim, center, width)
    
    '''CREATE MAGNETIC DISTRIBUTION'''
    mc.create_hom_mag(dim, res, beta, mag_shape, filename)
    
    '''LOAD MAGNETIC DISTRIBUTION'''
    mag_data = dl.MagDataLLG(filename)
    
    '''NUMERICAL SOLUTION'''
    # numerical solution Real Space (Slab):
    jacobi = np.zeros((dim[0]*dim[1], 2*dim[0]*dim[1]))
    tic = time.clock()
    pm.real_space(mag_data, 'slab', b_0, jacobi=jacobi)
    toc = time.clock()
    np.savetxt('../output/jacobi.npy', jacobi)
    print 'Time for Real Space Approach with Jacobi-Matrix (Slab): ' + str(toc - tic)
    
    return jacobi
    
    
if __name__ == "__main__":
    try:
        jacobi = phase_from_mag()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
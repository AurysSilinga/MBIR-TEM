# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 11:15:38 2013

@author: Jan
"""

import time
import pdb
import traceback
import sys
import os

import numpy as np
from numpy import pi

import pyramid.magcreator as mc
import pyramid.projector as pj
import pyramid.phasemapper as pm
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap


def get_jacobi():
    '''Calculate and display the phase map from a given magnetization.
    Arguments:
        None
    Returns:
        None

    '''
    directory = '../../output/reconstruction'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + '/jacobi.npy'
    # TODO: Input via GUI
    b_0 = 1.0  # in T
    dim = (1, 3, 3)  # in px (y,x)
    res = 10.0  # in nm
    phi = pi/4

    center = (0, int(dim[1]/2), int(dim[2]/2))  # in px (y,x) index starts with 0!
    width = (0, 1, 1)  # in px (y,x)

    mag_data = MagData(res, mc.create_mag_dist_homog(mc.Shapes.slab(dim, center, width), phi))
    projection = pj.simple_axis_projection(mag_data)

    '''NUMERICAL SOLUTION'''
    # numerical solution Real Space (Slab):
    jacobi = np.zeros((dim[2]*dim[1], 2*dim[2]*dim[1]))
    tic = time.clock()
    phase_map = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab', b_0, jacobi=jacobi))
    toc = time.clock()
    phase_map.display()
    np.savetxt(filename, jacobi)
    print 'Time for Real Space Approach with Jacobi-Matrix (Slab): ' + str(toc - tic)

    return jacobi


if __name__ == "__main__":
    try:
        jacobi = get_jacobi()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

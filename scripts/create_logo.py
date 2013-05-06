# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 11:15:38 2013

@author: Jan
"""

import pyramid.magcreator as mc
import pyramid.dataloader as dl
import pyramid.phasemap as pm
import pyramid.holoimage as hi
import pdb, traceback, sys
import numpy as np
from numpy import pi


def create_logo():
    '''Calculate and display the phase map from a given magnetization.
    Arguments:
        None
    Returns:
        None
    
    '''
    filename = '../output/mag_distr_logo.txt'
    b_0 = 1.0  # in T
    res = 10.0  # in nm
    beta = pi/2  # in rad
    density = 10
    dim = (128, 128)    
    
    x = range(dim[1])
    y = range(dim[0])
    xx, yy = np.meshgrid(x, y)    
    bottom = (yy >= 0.25*dim[0])
    left   = (yy <= 0.75/0.5 * dim[0]/dim[1] * xx)
    right  = np.fliplr(left)
    mag_shape = np.logical_and(np.logical_and(left, right), bottom)
    
    mc.create_hom_mag(dim, res, beta, mag_shape, filename)
    mag_data = dl.MagDataLLG(filename)
    phase= pm.real_space(mag_data, 'slab', b_0)  
    holo = hi.holo_image(phase, mag_data.res, density)
    hi.display_holo(holo, 'PYRAMID - LOGO')
    
    
if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description='Create the PYRAMID logo')
#    parser.add_argument('-d','--dimensions', help='Logo dimensions.', required=False)
#    args = parser.parse_args()
#    if args.dimensions is None:
#        args.dimensions = (128,128)
    try:
        create_logo()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
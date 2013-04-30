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
from numpy import pi


def create_logo():
    '''Calculate and display the phase map from a given magnetization.
    Arguments:
        None
    Returns:
        None
    
    '''

    filename = '../output/logo_magnetization.txt'
    b_0 = 1.0  # in T
    res = 10.0  # in nm
    beta = pi/2
    density = 10
    
    mc.create_logo(128, res, beta, filename)
    mag_data = dl.MagDataLLG(filename)
    phase= pm.real_space_slab(mag_data, b_0)  
    holo = hi.holo_image(phase, mag_data.res, density)
    hi.display_holo(holo, '')
    
    
if __name__ == "__main__":
    try:
        create_logo()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
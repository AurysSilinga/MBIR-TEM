# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 11:15:38 2013

@author: Jan
"""

import numpy as np
import pyramid.phasemap as pm
import pdb, traceback, sys


def compare_pixel_fields():
    '''Calculate and display the phase map from a given magnetization.
    Arguments:
        None
    Returns:
        None
    
    '''
    # TODO: Input via GUI
    b_0 = 1.0  # in T
    res = 10.0   
    dim = (10, 10)
    
    x_big = np.linspace(-(dim[1]-1), dim[1]-1, num=2*dim[1]-1)
    y_big = np.linspace(-(dim[0]-1), dim[0]-1, num=2*dim[0]-1)
    xx_big, yy_big = np.meshgrid(x_big, y_big)    
    
    phi_cos_real_slab = pm.phi_pixel('slab', xx_big, yy_big, res, b_0)
    pm.display_phase(phi_cos_real_slab, res, 'Phase of one Pixel-Slab (Cos - Part)')
    phi_cos_real_disc = pm.phi_pixel('disc', xx_big, yy_big, res, b_0)
    pm.display_phase(phi_cos_real_disc, res, 'Phase of one Pixel-Disc (Cos - Part)')
    phi_cos_diff = phi_cos_real_disc - phi_cos_real_slab
    pm.display_phase(phi_cos_diff, res, 'Phase of one Pixel-Disc (Cos - Part)')
    
    
if __name__ == "__main__":
    try:
        compare_pixel_fields()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
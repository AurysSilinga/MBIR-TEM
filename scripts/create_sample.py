# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 11:15:38 2013

@author: Jan
"""

import pyramid.magcreator as mc
import pdb, traceback, sys
from numpy import pi


def create_sample():
    '''Calculate and display the phase map from a given magnetization.
    Arguments:
        None
    Returns:
        None
    
    '''
    
    # TODO: Input via GUI
    key = 'slab'
    
    filename = '../output/mag_distr_' + key + '.txt'    
    dim = (50, 50)  # in px (y,x)
    res = 1.0  # in nm
    beta = pi/4
    plot_mag_distr = True    
    
    center = (24, 24)  # in px (y,x) index starts with 0!
    width  = (25, 25)  # in px (y,x)
    radius = 12.5  # in px
    pos = 24  # in px
    spacing = 5  # in px
    x_or_y = 'y'  
    pixel = (24, 24) # in px    
    
    if   key == 'slab':
        mag_shape = mc.slab(dim, center, width)
    elif key == 'disc':
        mag_shape = mc.disc(dim, center, radius)
    elif key == 'filament':
        mag_shape = mc.filament(dim, pos, x_or_y)
    elif key == 'alternating_filaments':
        mag_shape = mc.alternating_filaments(dim, spacing, x_or_y)
    elif key == 'pixel':
        mag_shape = mc.single_pixel(dim, pixel)
    
    mc.create_hom_mag(dim, res, beta, mag_shape, filename, plot_mag_distr)
    
    
if __name__ == "__main__":
    try:
        create_sample()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
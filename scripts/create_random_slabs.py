# -*- coding: utf-8 -*-
"""Create random magnetic distributions."""

import random as rnd
import pdb, traceback, sys
import numpy as np
from numpy import pi

import pyramid.magcreator as mc
from pyramid.magdata import MagData
import pyramid.phasemapper as pm
import pyramid.projector as pj
import pyramid.holoimage as hi
from pyramid.phasemap import PhaseMap


def create_random_slabs():
    '''Calculate, display and save a random magnetic distribution to file.
    Arguments:
        None
    Returns:
        None
    
    '''
    # Input parameters:
    count = 10
    dim = (1, 128, 128)  
    res = 10 # in nm
    rnd.seed(42)
    w_max = 10
    # Create lists for magnetic objects:
    mag_shape_list = np.zeros((count,) + dim)
    beta_list      = np.zeros(count) 
    magnitude_list = np.zeros(count)
    for i in range(count):
        width  = (1, rnd.randint(1, w_max), rnd.randint(1, w_max))
        center = (rnd.randrange(int(width[0]/2), dim[0]-int(width[0]/2)),
                  rnd.randrange(int(width[1]/2), dim[1]-int(width[1]/2)),
                  rnd.randrange(int(width[2]/2), dim[2]-int(width[2]/2)))
        mag_shape_list[i,...] = mc.Shapes.slab(dim, center, width)
        beta_list[i] = 2*pi*rnd.random()
        magnitude_list[i] = 1#rnd.random()
    # Create magnetic distribution:
    magnitude = mc.create_mag_dist_comb(mag_shape_list, beta_list, magnitude_list) 
    mag_data = MagData(res, magnitude)
    mag_data.quiver_plot()
    mag_data.save_to_llg('../output/mag_dist_random_slabs.txt')
    projection = pj.simple_axis_projection(mag_data)
    phase_map  = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab'))    
    hi.display(hi.holo_image(phase_map, 10))
    
    
if __name__ == "__main__":
    try:
        create_random_slabs()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
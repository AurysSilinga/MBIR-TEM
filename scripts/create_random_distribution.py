# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:05:40 2013

@author: Jan
"""

import random as rnd
import numpy as np
import pyramid.magcreator as mc
import pdb, traceback, sys
from numpy import pi


def create_random_distribution():
    
    count = 10
    dim = (1, 128, 128)    
    res = 10 # in nm
    
    rnd.seed(42)
    
    mag_shape_list = np.zeros((count, dim[0], dim[1], dim[2]))
    beta_list      = np.zeros(count) 
    magnitude_list = np.zeros(count)
    
    for i in range(count):
        pixel = (rnd.randrange(dim[0]), rnd.randrange(dim[1]), rnd.randrange(dim[2]))
        mag_shape_list[i,...] = mc.shape_single_pixel(dim, pixel)
        beta_list[i] = 2*pi*rnd.random()
        magnitude_list[i] = 1#rnd.random()
        
    mag_data = mc.create_mag_dist(dim, res, mag_shape_list, beta_list, magnitude_list)
    mag_data.quiver_plot()
    #mag_data.quiver_plot_3D()
    mag_data.save_to_llg('../output/mag_dist_random_pixel.txt')
    
    
if __name__ == "__main__":
    try:
        create_random_distribution()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
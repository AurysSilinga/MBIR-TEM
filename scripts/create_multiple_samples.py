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


def create_multiple_samples():
    '''Calculate, display and save a random magnetic distribution to file.
    Arguments:
        None
    Returns:
        None
    
    '''    
    filename = '../output/mag_dist_multiple_samples.txt'    
    res = 10.0  # nm
    dim = (64, 128, 128)
    # Slab:
    center = (32, 32, 32)  # in px (z, y, x), index starts with 0!
    width  = (48, 48, 48)  # in px (z, y, x)
    mag_shape_slab = mc.Shapes.slab(dim, center, width)
    # Disc:
    center = (32, 32, 96)  # in px (z, y, x), index starts with 0!
    radius = 24  # in px
    height =  24  # in px
    mag_shape_disc = mc.Shapes.disc(dim, center, radius, height)
    # Sphere:
    center = (32, 96, 64)  # in px (z, y, x), index starts with 0!
    radius = 24  # in px
    mag_shape_sphere = mc.Shapes.sphere(dim, center, radius)
    # Create lists for magnetic objects:
    mag_shape_list = [mag_shape_slab, mag_shape_disc, mag_shape_sphere]
    phi_list = [pi/4, pi/2, pi]
    # Create magnetic distribution:
    magnitude = mc.create_mag_dist_comb(mag_shape_list, phi_list) 
    mag_data = MagData(res, magnitude)
    mag_data.quiver_plot('z', dim[0]/2)
    mag_data.save_to_llg(filename)
    projection = pj.simple_axis_projection(mag_data)
    phase_map  = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab'))
    phase_map.display()
    hi.display(hi.holo_image(phase_map, 0.5))
    
    
    
if __name__ == "__main__":
    try:
        create_multiple_samples()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
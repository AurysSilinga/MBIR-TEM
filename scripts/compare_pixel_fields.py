# -*- coding: utf-8 -*-
"""Compare the phase map of one pixel for different real space approaches."""


import pdb, traceback, sys
from numpy import pi

import pyramid.magcreator  as mc
import pyramid.projector   as pj
import pyramid.phasemapper as pm
from pyramid.magdata  import MagData
from pyramid.phasemap import PhaseMap


def compare_pixel_fields():
    '''Calculate and display the phase map for different real space approaches.
    Arguments:
        None
    Returns:
        None
    
    '''
    # Input parameters:    
    res = 10.0  # in nm
    phi = pi/2  # in rad
    dim = (1, 11, 11)
    pixel = (0,  5,  5) 
    # Create magnetic data, project it, get the phase map and display the holography image:    
    mag_data   = MagData(res, mc.create_mag_dist(mc.Shapes.pixel(dim, pixel), phi)) 
    projection = pj.simple_axis_projection(mag_data)
    phase_map_slab = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab'))    
    phase_map_slab.display('Phase of one Pixel (Slab)')
    phase_map_disc = PhaseMap(res, pm.phase_mag_real(res, projection, 'disc'))    
    phase_map_disc.display('Phase of one Pixel (Disc)')
    phase_map_diff = PhaseMap(res, phase_map_disc.phase - phase_map_slab.phase)
    phase_map_diff.display('Phase difference of one Pixel (Disc - Slab)')
    
    
if __name__ == "__main__":
    try:
        compare_pixel_fields()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
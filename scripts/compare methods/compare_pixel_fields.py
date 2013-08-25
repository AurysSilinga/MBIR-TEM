#! python
# -*- coding: utf-8 -*-
"""Compare the phase map of one pixel for different real space approaches."""


import pdb
import traceback
import sys
import os

from numpy import pi

import pyramid.magcreator as mc
import pyramid.projector as pj
import pyramid.phasemapper as pm
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap


def compare_pixel_fields():
    '''Calculate and display the phase map for different real space approaches.
    Arguments:
        None
    Returns:
        None

    '''
    directory = '../../output/magnetic distributions'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Input parameters:
    res = 1.0  # in nm
    phi = pi/2  # in rad
    dim = (1, 101, 101)
    pixel = (0,  int(dim[1]/2),  int(dim[2]/2))
    limit = 0.25
    # Create magnetic data, project it, get the phase map and display the holography image:
    mag_data = MagData(res, mc.create_mag_dist_homog(mc.Shapes.pixel(dim, pixel), phi))
    mag_data.save_to_llg(directory + '/mag_dist_single_pixel.txt')
    projection = pj.simple_axis_projection(mag_data)
    phase_map_slab = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab'), 'mrad')
    phase_map_slab.display('Phase of one Pixel (Slab)', limit=limit)
    phase_map_disc = PhaseMap(res, pm.phase_mag_real(res, projection, 'disc'), 'mrad')
    phase_map_disc.display('Phase of one Pixel (Disc)', limit=limit)
    phase_map_diff = PhaseMap(res, phase_map_disc.phase - phase_map_slab.phase, 'mrad')
    phase_map_diff.display('Phase difference of one Pixel (Disc - Slab)')

if __name__ == "__main__":
    try:
        compare_pixel_fields()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

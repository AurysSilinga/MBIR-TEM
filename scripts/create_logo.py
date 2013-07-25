#! python
# -*- coding: utf-8 -*-
"""Create the Pyramid-Logo."""


import pdb
import traceback
import sys
import numpy as np
from numpy import pi

import pyramid.magcreator as mc
import pyramid.projector as pj
import pyramid.phasemapper as pm
import pyramid.holoimage as hi
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap


def create_logo():
    '''Calculate and display the Pyramid-Logo.
    Arguments:
        None
    Returns:
        None

    '''
    # Input parameters:
    res = 10.0  # in nm
    phi = -pi/2  # in rad
    density = 10
    dim = (1, 128, 128)
    # Create magnetic shape:
    mag_shape = np.zeros(dim)
    x = range(dim[2])
    y = range(dim[1])
    xx, yy = np.meshgrid(x, y)
    bottom = (yy >= 0.25*dim[1])
    left = (yy <= 0.75/0.5 * dim[1]/dim[2] * xx)
    right = np.fliplr(left)
    mag_shape[0, ...] = np.logical_and(np.logical_and(left, right), bottom)
    # Create magnetic data, project it, get the phase map and display the holography image:
    mag_data = MagData(res, mc.create_mag_dist(mag_shape, phi))
    projection = pj.simple_axis_projection(mag_data)
    phase_map = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab'))
    hi.display(hi.holo_image(phase_map, density), 'PYRAMID - LOGO')


if __name__ == "__main__":
    try:
        create_logo()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

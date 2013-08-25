# -*- coding: utf-8 -*-
"""Create random magnetic distributions."""


import random as rnd
import pdb
import traceback
import sys
import numpy as np
from numpy import pi

import pyramid.magcreator as mc
from pyramid.magdata import MagData
import pyramid.phasemapper as pm
import pyramid.projector as pj
import pyramid.holoimage as hi
from pyramid.phasemap import PhaseMap
import pyramid.reconstructor as rc


def reconstruct_random_distribution():
    '''Calculate and reconstruct a random magnetic distribution.
    Arguments:
        None
    Returns:
        None

    '''
    # Input parameters:
    n_pixel = 5
    dim = (1, 16, 16)
    b_0 = 1  # in T
    res = 10.0  # in nm
    rnd.seed(18)

    # Create empty MagData object and add random pixels:
    mag_data = MagData(res)
    for i in range(n_pixel):
        pixel = (rnd.randrange(dim[0]), rnd.randrange(dim[1]), rnd.randrange(dim[2]))
        mag_shape = mc.Shapes.pixel(dim, pixel)
        phi = 2 * pi * rnd.random()
        magnitude = rnd.random()
        mag_data.add_magnitude(mc.create_mag_dist_homog(mag_shape, phi, magnitude))
    # Plot magnetic distribution, phase map and holographic contour map:
    mag_data.quiver_plot()
    projection = pj.simple_axis_projection(mag_data)
    phase_map = PhaseMap(res, pm.phase_mag_real(res, projection, 'slab', b_0))
    hi.display_combined(phase_map, 10, 'Generated Distribution')

    # Reconstruct the magnetic distribution:
    mag_data_rec = rc.reconstruct_simple_leastsq(phase_map, mag_data.get_mask(), b_0)

    # Display the reconstructed phase map and holography image:
    projection_rec = pj.simple_axis_projection(mag_data_rec)
    phase_map_rec = PhaseMap(res, pm.phase_mag_real(res, projection_rec, 'slab', b_0))
    hi.display_combined(phase_map_rec, 10, 'Reconstructed Distribution')


if __name__ == "__main__":
    try:
        reconstruct_random_distribution()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

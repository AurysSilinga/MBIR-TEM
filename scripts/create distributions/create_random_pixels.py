#! python
# -*- coding: utf-8 -*-
"""Create random magnetic distributions."""

import random as rnd
import pdb
import traceback
import sys
import os

from numpy import pi

import pyramid.magcreator as mc
from pyramid.magdata import MagData
import pyramid.phasemapper as pm
import pyramid.projector as pj
import pyramid.holoimage as hi
from pyramid.phasemap import PhaseMap


def create_random_pixels():
    '''Calculate, display and save a random magnetic distribution to file.
    Arguments:
        None
    Returns:
        None

    '''
    directory = '../../output/magnetic distributions'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Input parameters:
    filename = directory + '/mag_dist_random_pixels.txt'
    # Input parameters:
    count = 10
    dim = (1, 128, 128)
    a = 10  # in nm
    rnd.seed(12)
    # Create empty MagData object and add pixels:
    mag_data = MagData(a)
    for i in range(count):
        pixel = (rnd.randrange(dim[0]), rnd.randrange(dim[1]), rnd.randrange(dim[2]))
        mag_shape = mc.Shapes.pixel(dim, pixel)
        phi = 2 * pi * rnd.random()
        magnitude = 1
        mag_data.add_magnitude(mc.create_mag_dist_homog(mag_shape, phi, magnitude=magnitude))
    # Plot magnetic distribution, phase map and holographic contour map:
    mag_data.quiver_plot()
    mag_data.save_to_llg(filename)
    projection = pj.simple_axis_projection(mag_data)
    phase_map = PhaseMap(a, pm.phase_mag(a, projection))
    hi.display(hi.holo_image(phase_map, 10))


if __name__ == "__main__":
    try:
        create_random_pixels()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

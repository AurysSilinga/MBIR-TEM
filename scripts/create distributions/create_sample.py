# -*- coding: utf-8 -*-
"""Create magnetic distributions with simple geometries."""


import pdb
import traceback
import sys
import os

from numpy import pi

import pyramid.magcreator as mc
from pyramid.magdata import MagData


def create_sample():
    '''Calculate, display and save simple magnetic distributions to file.
    Arguments:
        None
    Returns:
        None

    '''
    
    directory = '../../output/magnetic distributions'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Input parameters:
    key = 'sphere'
    filename = directory + '/mag_dist_' + key + '.txt'
    dim = (128, 128, 128)  # in px (z, y, x)
    res = 10.0  # in nm
    phi = pi/4
    # Geometry parameters:
    center = (64, 64, 64)  # in px (z, y, x), index starts with 0!
    width = (1, 50, 50)  # in px (z, y, x)
    radius = 25  # in px
    height = 1  # in px
    pos = (0, 63)  # in px (tuple of length 2)
    pixel = (0, 63, 63)  # in px (z, y, x), index starts with 0!
    # Determine the magnetic shape:
    if key == 'slab':
        mag_shape = mc.Shapes.slab(dim, center, width)
    elif key == 'disc':
        mag_shape = mc.Shapes.disc(dim, center, radius, height)
    elif key == 'sphere':
        mag_shape = mc.Shapes.sphere(dim, center, radius)
    elif key == 'filament':
        mag_shape = mc.Shapes.filament(dim, pos)
    elif key == 'pixel':
        mag_shape = mc.Shapes.pixel(dim, pixel)
    # Create magnetic distribution
    magnitude = mc.create_mag_dist_homog(mag_shape, phi)
    mag_data = MagData(res, magnitude)
    mag_data.quiver_plot()
    mag_data.save_to_llg(filename)


if __name__ == "__main__":
    try:
        create_sample()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

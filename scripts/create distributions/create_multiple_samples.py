#! python
# -*- coding: utf-8 -*-
"""Create random magnetic distributions."""


import os

from numpy import pi

import pyramid
import pyramid.magcreator as mc
from pyramid.magdata import MagData
from pyramid.phasemapper import PMConvolve
from pyramid.projector import SimpleProjector

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
directory = '../../output/magnetic distributions'
if not os.path.exists(directory):
    os.makedirs(directory)
# Input parameters:
filename = directory + '/mag_dist_multiple_samples.txt'
a = 10.0  # nm
dim = (64, 128, 128)
# Slab:
center = (32, 32, 32)  # in px (z, y, x), index starts with 0!
width = (48, 48, 48)  # in px (z, y, x)
mag_shape_slab = mc.Shapes.slab(dim, center, width)
# Disc:
center = (32, 32, 96)  # in px (z, y, x), index starts with 0!
radius = 24  # in px
height = 24  # in px
mag_shape_disc = mc.Shapes.disc(dim, center, radius, height)
# Sphere:
center = (32, 96, 64)  # in px (z, y, x), index starts with 0!
radius = 24  # in px
mag_shape_sphere = mc.Shapes.sphere(dim, center, radius)
# Create empty MagData object and add magnetized objects:
mag_data = MagData(a, mc.create_mag_dist_homog(mag_shape_slab, pi/4))
mag_data += MagData(a, mc.create_mag_dist_homog(mag_shape_disc, pi/2))
mag_data += MagData(a, mc.create_mag_dist_homog(mag_shape_sphere, pi))
# Plot the magnetic distribution, phase map and holographic contour map:
mag_data.quiver_plot()
mag_data.save_to_llg(filename)
phase_map = PMConvolve(a, SimpleProjector(dim))(mag_data)
phase_map.display_combined(density=0.5)

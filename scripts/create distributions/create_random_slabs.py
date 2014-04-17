#! python
# -*- coding: utf-8 -*-
"""Create random magnetic distributions."""


import random as rnd
import os

import numpy as np
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
filename = directory + '/mag_dist_random_pixels.txt'
count = 10
dim = (1, 128, 128)
a = 10  # in nm
rnd.seed(42)
w_max = 10
# Create empty MagData object and add slabs:
mag_data = MagData(a, np.zeros((3,)+dim))
for i in range(count):
    width = (1, rnd.randint(1, w_max), rnd.randint(1, w_max))
    center = (rnd.randrange(int(width[0]/2), dim[0]-int(width[0]/2)),
              rnd.randrange(int(width[1]/2), dim[1]-int(width[1]/2)),
              rnd.randrange(int(width[2]/2), dim[2]-int(width[2]/2)))
    mag_shape = mc.Shapes.slab(dim, center, width)
    phi = 2 * pi * rnd.random()
    magnitude = rnd.random()
    mag_data += MagData(a, mc.create_mag_dist_homog(mag_shape, phi, magnitude=magnitude))
# Plot magnetic distribution, phase map and holographic contour map:
mag_data.quiver_plot()
mag_data.save_to_llg(filename)
phase_map = PMConvolve(a, SimpleProjector(dim))(mag_data)
phase_map.display_combined(density=10)

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
from pyramid.phasemapper import pm

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
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
mag_data = MagData(a, np.zeros((3,)+dim))
for i in range(count):
    pixel = (rnd.randrange(dim[0]), rnd.randrange(dim[1]), rnd.randrange(dim[2]))
    mag_shape = mc.Shapes.pixel(dim, pixel)
    phi = 2 * pi * rnd.random()
    magnitude = 1
    mag_data += MagData(a, mc.create_mag_dist_homog(mag_shape, phi, magnitude=magnitude))
# Plot magnetic distribution, phase map and holographic contour map:
mag_data.quiver_plot()
mag_data.save_to_llg(filename)
phase_map = pm(mag_data)
phase_map.display_combined()

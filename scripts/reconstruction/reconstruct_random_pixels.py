# -*- coding: utf-8 -*-
"""Create random magnetic distributions."""


import os

import random as rnd

import numpy as np
from numpy import pi

import pyramid
import pyramid.magcreator as mc
from pyramid.magdata import MagData
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMConvolve
import pyramid.reconstruction as rc

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)

# Input parameters:
n_pixel = 5
dim = (1, 16, 16)
b_0 = 1  # in T
a = 10.0  # in nm
rnd.seed(18)

# Create empty MagData object and add random pixels:
mag_data = MagData(a, np.zeros((3,)+dim))
for i in range(n_pixel):
    pixel = (rnd.randrange(dim[0]), rnd.randrange(dim[1]), rnd.randrange(dim[2]))
    mag_shape = mc.Shapes.pixel(dim, pixel)
    phi = 2 * pi * rnd.random()
    magnitude = rnd.random()
    mag_data += MagData(a, mc.create_mag_dist_homog(mag_shape, phi, magnitude))
# Plot magnetic distribution, phase map and holographic contour map:
mag_data.quiver_plot()
phase_map = PMConvolve(a, SimpleProjector(dim), b_0)(mag_data)
phase_map.display_combined('Generated Distribution', density=10)

# Reconstruct the magnetic distribution:
mag_data_rec = rc.optimize_simple_leastsq(phase_map, mag_data.get_mask(), b_0)

# Display the reconstructed phase map and holography image:
phase_map_rec = PMConvolve(a, SimpleProjector(dim), b_0)(mag_data_rec)
phase_map_rec.display_combined('Reconstructed Distribution', density=10)

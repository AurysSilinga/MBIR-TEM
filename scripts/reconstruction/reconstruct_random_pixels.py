# -*- coding: utf-8 -*-
"""Create random magnetic distributions."""


import os

import random as rnd

import numpy as np
from numpy import pi

import pyramid
import pyramid.magcreator as mc
from pyramid.magdata import MagData
from pyramid.phasemapper import pm
from pyramid.dataset import DataSet
from pyramid.projector import SimpleProjector
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
phase_map = pm(mag_data)
phase_map.display_combined('Generated Distribution', gain=10)

# Reconstruct the magnetic distribution:

data = DataSet(a, dim, b_0, mag_data.get_mask())
data.append(phase_map, SimpleProjector(dim))
mag_data_rec, cost = rc.optimize_linear(data, max_iter=100)

# Display the reconstructed phase map and holography image:
phase_map_rec = pm(mag_data_rec)
phase_map_rec.display_combined('Reconstructed Distribution', gain=10)

#! python
# -*- coding: utf-8 -*-
"""Create the Pyramid-Logo."""


import os

import numpy as np
from numpy import pi

import pyramid
import pyramid.magcreator as mc
from pyramid.phasemapper import PMAdapterFM
from pyramid.magdata import MagData
from pyramid.projector import SimpleProjector

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
# Input parameters:
a = 10.0  # in nm
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
mag_data = MagData(a, mc.create_mag_dist_homog(mag_shape, phi))
mag_data.quiver_plot()
projector = SimpleProjector(dim)
phase_map = PMAdapterFM(a, projector)(mag_data)
phase_map.display_holo(density, 'PYRAMID - LOGO', interpolation='bilinear')

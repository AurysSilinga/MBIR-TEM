#! python
# -*- coding: utf-8 -*-
"""Create magnetic distribution of alternating filaments"""


import os

import numpy as np
from numpy import pi

import pyramid
import pyramid.magcreator as mc
from pyramid.magdata import MagData

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
directory = '../../output/magnetic distributions'
if not os.path.exists(directory):
    os.makedirs(directory)
# Input parameters:
filename = directory + '/mag_dist_alt_filaments.txt'
dim = (1, 21, 21)  # in px (z, y, x)
a = 10.0  # in nm
phi = pi/2
spacing = 5
# Create empty MagData object:
mag_data = MagData(a, np.zeros((3,)+dim))
count = int((dim[1]-1) / spacing) + 1
for i in range(count):
    pos = i * spacing
    mag_shape = mc.Shapes.filament(dim, (0, pos))
    mag_data += MagData(a, mc.create_mag_dist_homog(mag_shape, phi))
    phi *= -1  # Switch the angle
# Plot magnetic distribution
mag_data.quiver_plot()
mag_data.save_to_llg(filename)

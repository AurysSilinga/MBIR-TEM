#! python
# -*- coding: utf-8 -*-
"""Create a magnetic vortex distribution."""


import os

import matplotlib.pyplot as plt

import pyramid
import pyramid.magcreator as mc
from pyramid.phasemapper import pm
from pyramid.magdata import MagData

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
directory = '../../output/magnetic distributions'
if not os.path.exists(directory):
    os.makedirs(directory)
# Input parameters:
filename = directory + '/mag_dist_vortex.txt'
a = 10.0  # in nm
#    density = 1
dim = (32, 32, 32)
center = (int(dim[0]/2)-0.5, int(dim[1]/2)-0.5, int(dim[2]/2)-0.5)
print 'center', center
radius = 0.25 * dim[1]
height = dim[0]/4
# Create magnetic shape:
mag_shape = mc.Shapes.disc(dim, center, radius, height)
mag_data = MagData(a, mc.create_mag_dist_vortex(mag_shape, magnitude=0.75))
mag_data.quiver_plot()
mag_data.save_to_llg(filename)
phase_map = pm(mag_data)
phase_map.display_combined(gain=2)
phase_slice = phase_map.phase[dim[1]/2, :]
fig = plt.figure()
fig.add_subplot(111)
plt.plot(range(dim[1]), phase_slice)
mag_data.quiver_plot3d()

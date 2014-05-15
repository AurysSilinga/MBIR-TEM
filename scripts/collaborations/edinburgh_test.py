# -*- coding: utf-8 -*-
"""Created on Tue Jan 28 15:15:08 2014 @author: Jan"""


import os

import numpy as np

from matplotlib.ticker import FuncFormatter

import pyramid
from pyramid.magdata import MagData
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMAdapterFM

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)

# Load data:
data = np.loadtxt('../../output/data from Edinburgh/long_grain_remapped_0p0035.txt', delimiter=',')
# Set parameters:
a = 1000 * (data[1, 2] - data[0, 2])
dim = len(np.unique(data[:, 2])), len(np.unique(data[:, 1])), len(np.unique(data[:, 0]))
# Set magnetization:
mag_vec = np.concatenate([data[:, 3], data[:, 4], data[:, 5]])
x_mag = np.reshape(data[:, 3], dim, order='F')
y_mag = np.reshape(data[:, 4], dim, order='F')
z_mag = np.reshape(data[:, 5], dim, order='F')
magnitude = np.array((x_mag, y_mag, z_mag))
mag_data = MagData(a, magnitude)
# Pad and upscale:
mag_data.pad(30, 20, 0)
mag_data.scale_up()
# Phasemapping:
projector = SimpleProjector(mag_data.dim)
phasemapper = PMAdapterFM(mag_data.a, projector)
phase_map = phasemapper(mag_data)
# Plot:
phase_axis = phase_map.display_combined(density=20, interpolation='bilinear',
                                        grad_encode='bright')[0]
phase_axis.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:3.0f}'.format(x*mag_data.a)))
phase_axis.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:3.0f}'.format(x*mag_data.a)))

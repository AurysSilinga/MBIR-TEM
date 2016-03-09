# -*- coding: utf-8 -*-
"""Create magnetization distributions from fortran sorted txt-files."""


import os
import numpy as np
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
filename = 'long_grain_remapped_0p0070.txt'
###################################################################################################

# Load data:
data = np.loadtxt(os.path.join(py.DIR_FILES, 'txtfortran', filename), delimiter=',')
# Get parameters:
a = 1000 * (data[1, 2] - data[0, 2])
dim = len(np.unique(data[:, 2])), len(np.unique(data[:, 1])), len(np.unique(data[:, 0]))
# Get magnetization:
mag_vec = np.concatenate([data[:, 3], data[:, 4], data[:, 5]])
x_mag = np.reshape(data[:, 3], dim, order='F')
y_mag = np.reshape(data[:, 4], dim, order='F')
z_mag = np.reshape(data[:, 5], dim, order='F')
magnitude = np.array((x_mag, y_mag, z_mag))

# Create and save VectorData object:
mag_data = py.VectorData(a, magnitude)
mag_data.save_to_netcdf4('magdata_txtfortran_{}'.format(filename.replace('.txt', '.nc')))

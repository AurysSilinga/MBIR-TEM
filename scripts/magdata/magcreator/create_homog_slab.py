#! python
# -*- coding: utf-8 -*-
"""Create homogeneous slab."""

import logging.config

import numpy as np

import pyramid as pr

logging.config.fileConfig(pr.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (32, 32, 32)
a = 1.0  # in nm
phi = np.pi / 2  # in rad
theta = np.pi / 2  # in rad
amplitude = 1
filename = 'magdata_mc_homog_slab.hdf5'

# Magnetic shape:
center = (dim[0] // 2 - 0.5, dim[1] // 2 - 0.5, dim[2] // 2 - 0.5)
width = (dim[0] // 8, dim[1] // 2, dim[2] // 4)
mag_shape = pr.magcreator.Shapes.slab(dim, center, width)

# Create and save VectorData object:
mag_data = pr.VectorData(a, pr.magcreator.create_mag_dist_homog(mag_shape, phi, theta, amplitude))
mag_data.save_to_hdf5(filename, overwrite=True)
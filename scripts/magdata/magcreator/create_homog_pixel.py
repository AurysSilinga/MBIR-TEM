#! python
# -*- coding: utf-8 -*-
"""Create pyramid logo."""

import logging.config

import numpy as np

import pyramid as pr
import shapes

logging.config.fileConfig(pr.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (1, 8, 8)
a = 1.0  # in nm
phi = np.pi / 4  # in rad
theta = np.pi / 4  # in rad
amplitude = 1
filename = 'magdata_mc_pixel.hdf5'

# Magnetic shape:
pixel = (0, dim[1] // 4, dim[2] // 4)
mag_shape = shapes.Shapes.pixel(dim, pixel)

# Create and save VectorData object:
mag_data = pr.VectorData(a, pr.magcreator.create_mag_dist_homog(mag_shape, phi, theta, amplitude))
mag_data.save_to_hdf5(filename, overwrite=True)

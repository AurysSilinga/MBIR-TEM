#! python
# -*- coding: utf-8 -*-
"""Create pyramid logo."""


import numpy as np
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (1, 8, 8)
a = 1.0  # in nm
phi = np.pi/4  # in rad
theta = np.pi/4  # in rad
magnitude = 1
filename = 'magdata_mc_pixel.nc'

# Magnetic shape:
pixel = (0, dim[1]//4, dim[2]//4)
mag_shape = py.magcreator.Shapes.pixel(dim, pixel)

# Create and save VectorData object:
mag_data = py.VectorData(a, py.magcreator.create_mag_dist_homog(mag_shape, phi, theta, magnitude))
mag_data.save_to_netcdf4(filename)

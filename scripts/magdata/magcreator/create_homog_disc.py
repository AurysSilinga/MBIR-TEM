#! python
# -*- coding: utf-8 -*-
"""Create pyramid logo."""


import numpy as np
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (32, 32, 32)
a = 1.0  # in nm
phi = np.pi/4  # in rad
theta = np.pi/4  # in rad
magnitude = 1
filename = 'magdata_mc_homog_disc.nc'

# Magnetic shape:
center = (dim[0]//2-0.5, dim[1]//2-0.5, dim[2]//2-0.5)
radius = dim[2]//4
height = dim[0]//2
mag_shape = py.magcreator.Shapes.disc(dim, center, radius, height)

# Create and save MagData object:
mag_data = py.MagData(a, py.magcreator.create_mag_dist_homog(mag_shape, phi, theta, magnitude))
mag_data.save_to_netcdf4(filename)

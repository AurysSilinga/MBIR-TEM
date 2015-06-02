#! python
# -*- coding: utf-8 -*-
"""Create vortex disc magnetization distribution."""


import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (32, 32, 32)
a = 10.0  # in nm
axis = 'z'
magnitude = 1
filename = 'magdata_mc_vortex_sphere.nc'

# Magnetic shape:
center = (dim[0]//2-0.5, dim[1]//2-0.5, dim[2]//2-0.5)
radius = dim[2]//4  # in px
mag_shape = py.magcreator.Shapes.sphere(dim, center, radius)

# Create and save MagData object:
mag_data = py.MagData(a, py.magcreator.create_mag_dist_vortex(mag_shape, center, axis, magnitude))
mag_data.save_to_netcdf4(filename)

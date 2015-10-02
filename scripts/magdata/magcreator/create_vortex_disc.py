#! python
# -*- coding: utf-8 -*-
"""Create vortex disc magnetization distribution."""


import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (64, 64, 64)
a = 10.0  # in nm
axis = 'x'
magnitude = 1
filename = 'magdata_mc_vortex_disc_x.nc'

# Magnetic shape:
center = (dim[0]//2, dim[1]//2, dim[2]//2)
radius = dim[2]//4
height = dim[0]//2
mag_shape = py.magcreator.Shapes.disc(dim, center, radius, height, axis)

# Create and save MagData object:
mag_data = py.MagData(a, py.magcreator.create_mag_dist_vortex(mag_shape, center, axis, magnitude))
mag_data.save_to_netcdf4(filename)

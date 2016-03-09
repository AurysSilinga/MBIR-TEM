#! python
# -*- coding: utf-8 -*-
"""Create vortex disc magnetization distribution."""


import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (1, 512, 512)
a = 1.0  # in nm
axis = 'z'
magnitude = 1
filename = 'magdata_mc_paper2_vortex_slab_simulated.nc'

# Magnetic shape:
center = (0, dim[1]//2-0.5, dim[2]//2-0.5)
width = (0, 128, 128)  # in px
mag_shape = py.magcreator.Shapes.slab(dim, center, width)

# Create and save VectorData object:
mag_data = py.VectorData(a,
                         py.magcreator.create_mag_dist_vortex(mag_shape, center, axis, magnitude))
mag_data.save_to_netcdf4(filename)

#! python
# -*- coding: utf-8 -*-
"""Create vortex disc magnetization distribution."""

import logging.config

import pyramid as pr
import shapes

logging.config.fileConfig(pr.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (64, 64, 64)
a = 10.0  # in nm
axis = 'x'
amplitude = 1
filename = 'magdata_mc_vortex_disc_x.hdf5'

# Magnetic shape:
center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)
radius = dim[2] // 4
height = dim[0] // 2
mag_shape = shapes.Shapes.disc(dim, center, radius, height, axis)

# Create and save VectorData object:
magnitude = pr.magcreator.create_mag_dist_vortex(mag_shape, center, axis, amplitude)
mag_data = pr.VectorData(a, magnitude)
mag_data.save_to_hdf5(filename, overwrite=True)

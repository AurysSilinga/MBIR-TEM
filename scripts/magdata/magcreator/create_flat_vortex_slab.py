#! python
# -*- coding: utf-8 -*-
"""Create vortex disc magnetization distribution."""

import logging.config

import pyramid as pr
import shapes

logging.config.fileConfig(pr.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (1, 512, 512)
a = 1.0  # in nm
axis = 'z'
amplitude = 1
filename = 'magdata_mc_vortex_slab.hdf5'

# Magnetic shape:
center = (0, dim[1] // 2 - 0.5, dim[2] // 2 - 0.5)
width = (1, 128, 128)  # in px
mag_shape = shapes.Shapes.slab(dim, center, width)

# Create and save VectorData object:
magnitude = pr.magcreator.create_mag_dist_vortex(mag_shape, center, axis, amplitude)
mag_data = pr.VectorData(a, magnitude)
mag_data.save_to_hdf5(filename, overwrite=True)

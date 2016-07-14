#! python
# -*- coding: utf-8 -*-
"""Create pyramid logo."""

import numpy as np

import pyramid as pr


# Parameters:
dim = (1, 32, 32)
a = 1.0  # in nm
phi = np.pi / 2  # in rad
theta = np.pi / 2  # in rad
amplitude = 1
filename = 'magdata_mc_homog_filament.hdf5'

# Magnetic shape:
pos = (0, dim[1] // 2)
mag_shape = pr.shapes.filament(dim, pos)

# Create and save VectorData object:
mag_data = pr.VectorData(a, pr.magcreator.create_mag_dist_homog(mag_shape, phi, theta, amplitude))
mag_data.save_to_hdf5(filename, overwrite=True)

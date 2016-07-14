#! python
# -*- coding: utf-8 -*-
"""Create pyramid logo."""

import numpy as np

import pyramid as pr


# Parameters:
dim = (1, 256, 256)
a = 1.0  # in nm
phi = -np.pi / 2  # in rad
theta = np.pi / 2  # in rad
amplitude = 1
filename = 'magdata_mc_pyramid_logo.hdf5'

# Magnetic shape:
mag_shape = np.zeros(dim)
x = range(dim[2])
y = range(dim[1])
xx, yy = np.meshgrid(x, y)
bottom = (yy >= 0.25 * dim[1])
left = (yy <= 0.75 / 0.5 * dim[1] / dim[2] * xx)
right = np.fliplr(left)
mag_shape[0, ...] = np.logical_and(np.logical_and(left, right), bottom)

# Create and save VectorData object:
mag_data = pr.VectorData(a, pr.magcreator.create_mag_dist_homog(mag_shape, phi, theta, amplitude))
mag_data.save_to_hdf5(filename, overwrite=True)

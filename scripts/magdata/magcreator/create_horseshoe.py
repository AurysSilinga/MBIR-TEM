#! python
# -*- coding: utf-8 -*-
"""Create magnetic horseshoe vortex."""

import numpy as np

import pyramid as pr


# Parameters:
dim = (32, 128, 128)
a = 10.0  # in nm
center = (dim[0] // 2 - 0.5, dim[1] // 2 - 0.5, dim[2] // 2 - 0.5)
radius_core = dim[1] // 8
radius_shell = dim[1] // 4
height = dim[0] // 2
filename = 'magdata_mc_horseshoe.hdf5'

# Magnetic shape:
mag_shape_core = pr.shapes.disc(dim, center, radius_core, height)
mag_shape_outer = pr.shapes.disc(dim, center, radius_shell, height)
mag_shape_horseshoe = np.logical_xor(mag_shape_outer, mag_shape_core)
mag_shape_horseshoe[:, dim[1] // 2:, :] = False

# Create and save VectorData object:
mag_data = pr.VectorData(a, pr.magcreator.create_mag_dist_vortex(mag_shape_horseshoe))
mag_data.save_to_hdf5(filename, overwrite=True)

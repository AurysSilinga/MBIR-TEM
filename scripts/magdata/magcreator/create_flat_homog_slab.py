#! python
# -*- coding: utf-8 -*-
"""Create vortex disc magnetization distribution."""

import numpy as np

import pyramid as pr


# Parameters:
dim = (1, 512, 512)
a = 1.0  # in nm
axis = 'z'
amplitude = 1
filename = 'magdata_mc_flat_homog_slab.hdf5'
rotation = np.pi / 4
tilt = 0

# Magnetic shape:
center_1 = (0, dim[1] // 2 - 0.5 + 40, dim[2] // 2 - 0.5)
center_2 = (0, dim[1] // 2 - 0.5 - 40, dim[2] // 2 - 0.5)
width = (1, 80, 80)  # in px
mag_shape_1 = pr.shapes.slab(dim, center_1, width)
mag_shape_2 = pr.shapes.slab(dim, center_2, width)

# Create and save VectorData object:
mag = pr.magcreator.create_mag_dist_homog(mag_shape_1, phi=7 / 12. * np.pi, amplitude=amplitude)
mag += pr.magcreator.create_mag_dist_homog(mag_shape_2, phi=1 / 3. * np.pi, amplitude=amplitude)
mag_data = pr.VectorData(a, mag)
projector = pr.RotTiltProjector(mag_data.dim, rotation, tilt)
mag_proj = projector(mag_data)
mag_proj.save_to_hdf5(filename, overwrite=True)

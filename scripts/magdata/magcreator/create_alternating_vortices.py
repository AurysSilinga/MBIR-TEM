#! python
# -*- coding: utf-8 -*-
"""Create magnetic distribution of alternating filaments."""

import numpy as np
import pyramid as pr


# Parameters:
segments = 8
size = 32
segment_height = size / 4
dim = (segment_height * (segments + 2), size, size)
a = 1.
filename = 'magdata_mc_alternating_vortices.hdf5'

# Create and save VectorData object:
mag_data = pr.VectorData(a, np.zeros((3,) + dim))
for i in range(segments):
    axis = 'z' if i % 2 == 0 else '-z'
    center = (segment_height * (i + 1 + 0.5), size / 2, size / 2)
    radius = size / 4
    height = segment_height
    mag_shape = pr.shapes.disc(dim, center=center, radius=radius, height=height)
    mag_amp = pr.magcreator.create_mag_dist_vortex(mag_shape=mag_shape, center=center, axis=axis)
    mag_data += pr.VectorData(1, mag_amp)
mag_data.save_to_hdf5(filename, overwrite=True)

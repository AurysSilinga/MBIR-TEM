#! python
# -*- coding: utf-8 -*-
"""Create magnetic distribution of alternating filaments."""

import logging.config

import numpy as np

import pyramid as pr

logging.config.fileConfig(pr.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (1, 21, 21)
a = 1.0  # in nm
phi = np.pi / 2
theta = np.pi / 2  # in rad
spacing = 5
filename = 'magdata_mc_alternating_filaments_spacing={}.hdf5'.format(spacing)

# Create and save VectorData object:
mag_data = pr.VectorData(a, np.zeros((3,) + dim))
count = int((dim[1] - 1) / spacing) + 1
for i in range(count):
    pos = i * spacing
    mag_shape = pr.magcreator.Shapes.filament(dim, (0, pos))
    mag_data += pr.VectorData(a, pr.magcreator.create_mag_dist_homog(mag_shape, phi))
    phi *= -1  # Switch the angle
mag_data.save_to_hdf5(filename, overwrite=True)

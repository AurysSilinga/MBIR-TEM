#! python
# -*- coding: utf-8 -*-
"""Create random magnetic distributions."""

import logging.config
import random as rnd

import numpy as np

import pyramid as pr

logging.config.fileConfig(pr.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (1, 128, 128)
a = 10  # in nm
count = 10
w_max = 10
rnd.seed(42)
filename = 'magdata_mc_{}_random_slabs.hdf5'.format(count)

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
mag_data = pr.VectorData(a, np.zeros((3,) + dim))
for i in range(count):
    width = (1, rnd.randint(1, w_max), rnd.randint(1, w_max))
    center = (rnd.randrange(int(width[0] / 2), dim[0] - int(width[0] / 2)),
              rnd.randrange(int(width[1] / 2), dim[1] - int(width[1] / 2)),
              rnd.randrange(int(width[2] / 2), dim[2] - int(width[2] / 2)))
    mag_shape = pr.magcreator.Shapes.slab(dim, center, width)
    phi = 2 * np.pi * rnd.random()
    mag_data += pr.VectorData(a, pr.magcreator.create_mag_dist_homog(mag_shape, phi))
mag_data.save_to_hdf5(filename, overwrite=True)
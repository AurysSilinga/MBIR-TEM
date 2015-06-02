#! python
# -*- coding: utf-8 -*-
"""Create random magnetic distributions."""


import numpy as np
import random as rnd
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (1, 128, 128)
a = 10  # in nm
count = 10
rnd.seed(12)
filename = 'magdata_mc_{}_random_pixels.nc'.format(count)

# Magnetic shape:
mag_shape = np.zeros(dim)
x = range(dim[2])
y = range(dim[1])
xx, yy = np.meshgrid(x, y)
bottom = (yy >= 0.25*dim[1])
left = (yy <= 0.75/0.5 * dim[1]/dim[2] * xx)
right = np.fliplr(left)
mag_shape[0, ...] = np.logical_and(np.logical_and(left, right), bottom)

# Create and save MagData object:
mag_data = py.MagData(a, np.zeros((3,)+dim))
for i in range(count):
    pixel = (rnd.randrange(dim[0]), rnd.randrange(dim[1]), rnd.randrange(dim[2]))
    mag_shape = py.magcreator.Shapes.pixel(dim, pixel)
    phi = 2 * np.pi * rnd.random()
    mag_data += py.MagData(a, py.magcreator.create_mag_dist_homog(mag_shape, phi))
mag_data.save_to_netcdf4(filename)

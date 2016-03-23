#! python
# -*- coding: utf-8 -*-
"""Create multiple magnetic distributions."""

import logging.config

import numpy as np

import pyramid as pr

logging.config.fileConfig(pr.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (64, 128, 128)
a = 10.0  # in nm
filename = 'magdata_mc_array_sphere_disc_slab.hdf5'

# Slab:
center = (32, 32, 32)  # in px (z, y, x), index starts with 0!
width = (48, 48, 48)  # in px (z, y, x)
mag_shape_slab = pr.magcreator.Shapes.slab(dim, center, width)
# Disc:
center = (32, 32, 96)  # in px (z, y, x), index starts with 0!
radius = 24  # in px
height = 24  # in px
mag_shape_disc = pr.magcreator.Shapes.disc(dim, center, radius, height)
# Sphere:
center = (32, 96, 64)  # in px (z, y, x), index starts with 0!
radius = 24  # in px
mag_shape_sphere = pr.magcreator.Shapes.sphere(dim, center, radius)

# Create and save VectorData object:
mag_data = pr.VectorData(a, pr.magcreator.create_mag_dist_homog(mag_shape_slab, np.pi / 4))
mag_data += pr.VectorData(a, pr.magcreator.create_mag_dist_homog(mag_shape_disc, np.pi / 2))
mag_data += pr.VectorData(a, pr.magcreator.create_mag_dist_homog(mag_shape_sphere, np.pi))
mag_data.save_to_hdf5(filename, overwrite=True)

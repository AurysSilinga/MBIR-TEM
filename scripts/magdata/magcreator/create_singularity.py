# -*- coding: utf-8 -*-
"""magnetic singularity"""

import logging.config

import numpy as np

import pyramid as pr

logging.config.fileConfig(pr.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (5, 5, 5)
center = (2., 2., 2.)
a = 1.0  # in nm
filename = 'magdata_mc_singularity.hdf5'

zz, yy, xx = np.indices(dim)
magnitude = np.array((xx - center[2], yy - center[1], zz - center[0]))
magnitude /= np.sqrt((magnitude ** 2).sum(axis=0))

# Create and save VectorData object:
mag_data = pr.VectorData(a, magnitude)
mag_data.save_to_hdf5(filename, overwrite=True)

# -*- coding: utf-8 -*-
"""magnetic singularity"""


import numpy as np
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (5, 5, 5)
center = (2., 2., 2.)
a = 1.0  # in nm
filename = 'magdata_mc_singularity.nc'

magnitude = np.zeros((3,) + dim)
zz, yy, xx = np.indices(dim)
magnitude = np.array((xx-center[2], yy-center[1], zz-center[0]))
magnitude /= np.sqrt((magnitude**2).sum(axis=0))

# Create and save VectorData object:
mag_data = py.VectorData(a, magnitude)
mag_data.quiver_plot3d(coloring='full angle')
mag_data.save_to_netcdf4(filename)

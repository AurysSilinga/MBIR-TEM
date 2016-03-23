# -*- coding: utf-8 -*-
"""Create magnetization distributions from fortran sorted txt-files."""

import logging.config
import os

import numpy as np

import pyramid as py

import matplotlib.pyplot as plt

logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
filename = 'J=1.D=0.084.H=0.0067.Bobber.dat'
scale = 1
###################################################################################################

path = os.path.join(py.DIR_FILES, 'dat', filename)
data = np.genfromtxt(path, dtype=np.float32, delimiter=',', usecols=(0, 1, 2, 3, 4, 5))
x, y, z, xmag, ymag, zmag = data.T
a = (y[1] - y[0]) * scale
dim_z = len(np.unique(z))
dim_y = len(np.unique(y))
dim_x = len(np.unique(x))
dim = (dim_z, dim_x, dim_y)  # Order of descending variance!
xmag = xmag.reshape(dim).swapaxes(1, 2)
ymag = ymag.reshape(dim).swapaxes(1, 2)
zmag = zmag.reshape(dim).swapaxes(1, 2)
magnitude = np.array((xmag, ymag, zmag))
mag_data = py.VectorData(a, magnitude)
mag_data.save_to_hdf5('magdata_dat_{}'.format(filename.replace('.dat', '.hdf5')), overwrite=True)
mag_data.quiver_plot3d(ar_dens=4, coloring='amplitude')
mag_data.quiver_plot3d(ar_dens=4, coloring='angle')
py.pm(mag_data).display_combined(interpolation='bilinear')
plt.show()

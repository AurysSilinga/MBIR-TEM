#! python
# -*- coding: utf-8 -*-
"""Create vortex disc magnetization distribution."""


import numpy as np
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

# Parameters:
dim = (1, 512, 512)
a = 1.0  # in nm
axis = 'z'
magnitude = 1
filename = 'magdata_mc_paper2_homog_slab_simulated.nc'
rotation = np.pi/4
tilt = 0

# Magnetic shape:
center_1 = (0, dim[1]//2-0.5+40, dim[2]//2-0.5)
center_2 = (0, dim[1]//2-0.5-40, dim[2]//2-0.5)
width = (0, 80, 80)  # in px
mag_shape_1 = py.magcreator.Shapes.slab(dim, center_1, width)
mag_shape_2 = py.magcreator.Shapes.slab(dim, center_2, width)

# Create and save MagData object:
mag = py.magcreator.create_mag_dist_homog(mag_shape_1, phi=7/12.*np.pi, magnitude=magnitude)
mag += py.magcreator.create_mag_dist_homog(mag_shape_2, phi=1/3.*np.pi, magnitude=magnitude)
mag_data = py.MagData(a, mag)
projector = py.RotTiltProjector(mag_data.dim, rotation, tilt)
mag_data = projector(mag_data)
mag_data.save_to_netcdf4(filename)
mag_data.quiver_plot(ar_dens=16)

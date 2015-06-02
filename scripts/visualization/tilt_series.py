# -*- coding: utf-8 -*-
"""Reconstruct a magnetization distributions from a single phase map."""


import os
import numpy as np
import matplotlib.pyplot as plt
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
mag_name = 'magdata_vtk_tube_160x30x1100nm'
dim_uv = (400, 150)
angles = np.linspace(-60, 60, num=7)
axis = 'x'
b_0 = 1
save_images = True
interpolation = 'bilinear'
gain = 'auto'
dpi = 300
###################################################################################################

# Load magnetization distribution:
mag_data = py.MagData.load_from_netcdf4(mag_name+'.nc').rot90('x').flip('x')
dim = mag_data.dim

# Construct data set and regularisator:
data = py.DataSet(mag_data.a, mag_data.dim, b_0)

# Calculate phase maps:
for angle in angles:
    angle_rad = angle * np.pi/180.
    if axis == 'x':
        projector = py.XTiltProjector(mag_data.dim, angle_rad, dim_uv)
    if axis == 'y':
        projector = py.YTiltProjector(mag_data.dim, angle_rad, dim_uv)
    print projector.get_info()
    phasemapper = py.PhaseMapperRDFC(py.Kernel(mag_data.a, projector.dim_uv, b_0))
    phase_map = phasemapper(projector(mag_data))
    im = phase_map.display_combined('({})'.format(projector.get_info(verbose=True)),
                                    gain=gain, interpolation=interpolation)
    if save_images:
        filename = '{}_{}'.format(mag_name.replace('magdata', 'phasemap'), projector.get_info())
        directory = os.path.join(py.DIR_FILES, 'images', 'tilt_series_{}'.format(mag_name))
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, filename)
        plt.savefig(filename+'.png', dpi=dpi)

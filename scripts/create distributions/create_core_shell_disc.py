#! python
# -*- coding: utf-8 -*-
"""Create a core-shell disc."""


import os

import numpy as np

import matplotlib.pyplot as plt

import pyramid
import pyramid.magcreator as mc
from pyramid.phasemapper import pm
from pyramid.magdata import MagData

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
directory = '../../output/magnetic distributions'
if not os.path.exists(directory):
    os.makedirs(directory)
# Input parameters:
filename = directory + '/mag_dist_core_shell_disc.txt'
a = 1.0  # in nm
dim = (32, 32, 32)
center = (dim[0]/2-0.5, int(dim[1]/2)-0.5, int(dim[2]/2)-0.5)
radius_core = dim[1]/8
radius_shell = dim[1]/4
height = dim[0]/2
# Create magnetic shape:
mag_shape_core = mc.Shapes.disc(dim, center, radius_core, height)
mag_shape_outer = mc.Shapes.disc(dim, center, radius_shell, height)
mag_shape_shell = np.logical_xor(mag_shape_outer, mag_shape_core)
mag_data = MagData(a, mc.create_mag_dist_vortex(mag_shape_shell, magnitude=0.75))
mag_data += MagData(a, mc.create_mag_dist_homog(mag_shape_core, phi=0, theta=0))
mag_data.quiver_plot('z-projection', proj_axis='z')
mag_data.quiver_plot('y-projection', proj_axis='y')
mag_data.save_to_llg(filename)
mag_data.quiver_plot3d()
phase_map_z = pm(mag_data)
phase_map_x = pm(mag_data)
phase_map_z.display_holo('Core-Shell structure (z-proj.)', gain=1E2)
phase_axis, holo_axis = phase_map_x.display_combined('Core-Shell structure (x-proj.)', gain=1E3)
phase_axis.set_xlabel('y [nm]')
phase_axis.set_ylabel('z [nm]')
holo_axis.set_xlabel('y-axis [px]')
holo_axis.set_ylabel('z-axis [px]')
phase_slice_z = phase_map_z.phase[dim[1]/2, :]
phase_slice_x = phase_map_x.phase[dim[0]/2, :]
fig = plt.figure()
fig.add_subplot(1, 1, 1)
plt.plot(range(dim[2]), phase_slice_z)
plt.title('Phase slice along x for z-projection')
fig = plt.figure()
fig.add_subplot(1, 1, 1)
plt.plot(range(dim[1]), phase_slice_x)
plt.title('Phase slice along y for x-projection')

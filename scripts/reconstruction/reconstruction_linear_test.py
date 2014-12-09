# -*- coding: utf-8 -*-
"""Created on Fri Feb 28 14:25:59 2014 @author: Jan"""


import os

import numpy as np
from numpy import pi

import pyramid
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap
from pyramid.projector import YTiltProjector, XTiltProjector
from pyramid.dataset import DataSet
from pyramid.regularisator import FirstOrderRegularisator
import pyramid.magcreator as mc
import pyramid.reconstruction as rc

from jutil.taketime import TakeTime

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)

###################################################################################################
print('--Generating input phase_maps')

a = 10.
b_0 = 1.
dim = (32, 32, 32)
dim_uv = dim[1:3]
count = 16
lam = 1E-4
use_fftw = True
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

shape = mc.Shapes.disc(dim, center, radius_shell, height)
magnitude = mc.create_mag_dist_vortex(shape)
mag_data = MagData(a, magnitude)

#mag_data.quiver_plot('z-projection', proj_axis='z')
#mag_data.quiver_plot('y-projection', proj_axis='y')
#mag_data.quiver_plot3d('Original distribution')

tilts_full = np.linspace(-pi/2, pi/2, num=count/2, endpoint=False)
tilts_miss = np.linspace(-pi/3, pi/3, num=count/2, endpoint=False)

projectors_xy_full, projectors_x_full, projectors_y_full = [], [], []
projectors_xy_miss, projectors_x_miss, projectors_y_miss = [], [], []
projectors_x_full.extend([XTiltProjector(dim, tilt) for tilt in tilts_full])
projectors_y_full.extend([YTiltProjector(dim, tilt) for tilt in tilts_full])
projectors_xy_full.extend(projectors_x_full)
projectors_xy_full.extend(projectors_y_full)
projectors_x_miss.extend([XTiltProjector(dim, tilt) for tilt in tilts_miss])
projectors_y_miss.extend([YTiltProjector(dim, tilt) for tilt in tilts_miss])
projectors_xy_miss.extend(projectors_x_miss)
projectors_xy_miss.extend(projectors_y_miss)

###################################################################################################
projectors = projectors_xy_miss
noise = 0
###################################################################################################
print('--Setting up data collection')

mask = mag_data.get_mask()
data = DataSet(a, dim, b_0, mask, use_fftw=use_fftw)
data.projectors = projectors
data.phase_maps = data.create_phase_maps(mag_data)

if noise != 0:
    for i, phase_map in enumerate(data.phase_maps):
        phase_map += PhaseMap(a, np.random.normal(0, noise, dim_uv))
        data.phase_maps[i] = phase_map

###################################################################################################
print('--Reconstruction')

reg = FirstOrderRegularisator(mask, lam, p=2)
info = []
with TakeTime('reconstruction'):
    mag_data_opt = rc.optimize_linear(data, regularisator=reg, max_iter=100, info=info)

###################################################################################################
print('--Plot stuff')

mag_data_opt.quiver_plot3d('Reconstructed distribution')
#(mag_data_opt - mag_data).quiver_plot3d('Difference')
#phase_maps_opt = data.create_phase_maps(mag_data_opt)

# TODO: iterations in jutil is one to many!

# -*- coding: utf-8 -*-
"""Created on Fri Feb 28 14:25:59 2014 @author: Jan"""


import os

import numpy as np
from numpy import pi

from time import clock

import pyramid
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap
from pyramid.projector import YTiltProjector, XTiltProjector
from pyramid.dataset import DataSet
from pyramid.regularisator import ZeroOrderRegularisator
import pyramid.magcreator as mc
import pyramid.reconstruction as rc

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)

###################################################################################################
print('--Generating input phase_maps')

a = 10.
b_0 = 1.
dim = (8, 32, 32)
dim_uv = dim[1:3]
count = 16

center = (int(dim[0]/2)-0.5, int(dim[1]/2)-0.5, int(dim[2]/2)-0.5)
mag_shape = mc.Shapes.disc(dim, center, dim[2]/4, dim[2]/2)
magnitude = mc.create_mag_dist_vortex(mag_shape)
mag_data = MagData(a, magnitude)

#vortex_shape = mc.Shapes.disc(dim, (3.5, 9.5, 9.5), 5, 4)
#sphere_shape = mc.Shapes.sphere(dim, (3.5, 22.5, 9.5), 3)
#slab_shape = mc.Shapes.slab(dim, (3.5, 15.5, 22.5), (5, 8, 8))
#mag_data = MagData(a, mc.create_mag_dist_vortex(vortex_shape, (3.5, 9.5, 9.5)))
#mag_data += MagData(a, mc.create_mag_dist_homog(sphere_shape, pi/4, pi/4))
#mag_data += MagData(a, mc.create_mag_dist_homog(slab_shape, -pi/6))

mag_data.quiver_plot3d()

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
projectors = projectors_xy_full
noise = 0.0
###################################################################################################
print('--Setting up data collection')

data = DataSet(a, dim, b_0)
data.projectors = projectors
data.phase_maps = data.create_phase_maps(mag_data)

if noise != 0:
    for phase_map in data.phase_maps:
        phase_map += PhaseMap(a, np.random.normal(0, noise, dim_uv))

###################################################################################################
print('--Test simple solver')

lam = 1E-4
reg = ZeroOrderRegularisator(lam)
start = clock()
mag_data_opt = rc.optimize_sparse_cg(data, regularisator=reg, maxiter=100, verbosity=2)
print 'Time:', str(clock()-start)
mag_data_opt.quiver_plot3d()
(mag_data_opt - mag_data).quiver_plot3d()

###################################################################################################
print('--Plot stuff')

phase_maps_opt = data.create_phase_maps(mag_data_opt)
#data.display_phase()
#data.display_phase(phase_maps_opt)
phase_diffs = [(data.phase_maps[i]-phase_maps_opt[i]) for i in range(len(data.phase_maps))]
[phase_diff.display_phase() for phase_diff in phase_diffs]

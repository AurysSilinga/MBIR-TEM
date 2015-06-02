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
dim = (64, 64, 64)
dim_uv = dim[1:3]
count = 16
lam = 1E-4
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
data = DataSet(a, dim, b_0, mask)
data.projectors = projectors
data.phase_maps = data.create_phase_maps(mag_data)

if noise != 0:
    for i, phase_map in enumerate(data.phase_maps):
        phase_map += PhaseMap(a, np.random.normal(0, noise, dim_uv))
        data.phase_maps[i] = phase_map

###################################################################################################
print('--Reconstruction')

reg = FirstOrderRegularisator(mask, lam, p=2)
with TakeTime('reconstruction'):
    mag_data_opt, cost = rc.optimize_linear(data, regularisator=reg, max_iter=100)
print 'Cost:', cost.chisq

###################################################################################################
#print('--Plot stuff')
#
#limit = 1.2
#mag_data.quiver_plot3d('Original distribution', limit=limit)
#mag_data_opt.quiver_plot3d('Reconstructed distribution', limit=limit)
#(mag_data_opt - mag_data).quiver_plot3d('Difference')
#phase_maps_opt = data.create_phase_maps(mag_data_opt)
#
#
#from pyramid.diagnostics import Diagnostics
#from matplotlib.patches import Rectangle
#
#
#diag = Diagnostics(mag_data_opt.mag_vec, cost, max_iter=2000)
#
#print 'position:', diag.pos
#print 'std:', diag.std
#gain_maps = diag.get_gain_row_maps()
#axis, cbar = gain_maps[count//2].display_phase()
#axis.add_patch(Rectangle((diag.pos[3], diag.pos[2]), 1, 1, linewidth=2, color='g', fill=False))
#cbar.set_label(u'magnetization/phase [1/rad]', fontsize=15)
#diag.get_avg_kern_row().quiver_plot3d()
#mcon = diag.measure_contribution
#print 'measurement contr. (min - max): {:.2f} - {:.2f}'.format(mcon.min(), mcon.max())
#px_avrg, fwhm, (x, y, z) = diag.calculate_averaging()
#print 'px_avrg:', px_avrg
#print 'fwhm:', fwhm
#
#diag.pos = (1, dim[0]//2, dim[1]//2, dim[2]//2)
#print 'position:', diag.pos
#print 'std:', diag.std
#gain_maps = diag.get_gain_row_maps()
#axis, cbar = gain_maps[count//2].display_phase()
#axis.add_patch(Rectangle((diag.pos[3], diag.pos[2]), 1, 1, linewidth=2, color='g', fill=False))
#cbar.set_label(u'magnetization/phase [1/rad]', fontsize=15)
#diag.get_avg_kern_row().quiver_plot3d()
#mcon = diag.measure_contribution
#print 'measurement contr. (min - max): {:.2f} - {:.2f}'.format(mcon.min(), mcon.max())
#px_avrg, fwhm, (x, y, z) = diag.calculate_averaging()
#print 'px_avrg:', px_avrg
#print 'fwhm:', fwhm

# -*- coding: utf-8 -*-
"""Created on Fri Feb 28 14:25:59 2014 @author: Jan"""


import os

import numpy as np
from numpy import pi

import itertools

from mayavi import mlab

from pyramid import *  #analysis:ignore
import pyramid.magcreator as mc
import pyramid.reconstruction as rc

from jutil.taketime import TakeTime

import psutil
import gc

import shelve

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')
PATH = '../../output/3d reconstruction/Phase 1/'

logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
if not os.path.exists(PATH):
    os.mkdir(PATH)
proc = psutil.Process(os.getpid())
###################################################################################################
print('--Constant parameters')

a = 10.
b_0 = 1.

###################################################################################################
print('--Magnetization distributions')

dim = (32, 32, 32)
center = (dim[0]/2-0.5, dim[1]/2-0.5, dim[2]/2-0.5)
radius_core = dim[1]/8
radius_shell = dim[1]/4
height = dim[0]/2
width_cube = (dim[0]/4, dim[1]/4, dim[2]/4)
width_slab = (dim[0]/4, dim[1]/4, dim[2]/2)
# Vortex:
mag_shape_disc = mc.Shapes.disc(dim, center, radius_shell, height)
mag_data_vortex = MagData(a, mc.create_mag_dist_vortex(mag_shape_disc))
# Core-Shell:
mag_shape_core = mc.Shapes.disc(dim, center, radius_core, height)
mag_shape_shell = np.logical_xor(mag_shape_disc, mag_shape_core)
mag_data_core_shell = MagData(a, mc.create_mag_dist_vortex(mag_shape_shell, magnitude=0.75))
mag_data_core_shell += MagData(a, mc.create_mag_dist_homog(mag_shape_core, phi=0, theta=0))
# Cubic slab:
mag_shape_cube = mc.Shapes.slab(dim, center, width_cube)
mag_data_cube = MagData(a, mc.create_mag_dist_homog(mag_shape_cube, 0))
# Rectangular slab:
mag_shape_slab = mc.Shapes.slab(dim, center, width_slab)
mag_data_slab = MagData(a, mc.create_mag_dist_homog(mag_shape_slab, 0))

###################################################################################################
print('--Set up configurations')

mag_datas = {'rect_slab': mag_data_slab}
masks = {'mask': True, 'nomask': False}
xy_tilts = {'y': [False, True]}
max_tilts = [60]
tilt_steps = [10]
noises = [0]
orders = [1]
lambdas = [1E-2, 1E-3, 1E-4, 1E-5, 1E-6]
# Combining everything:
config_list = [mag_datas.keys(), masks.keys(), xy_tilts.keys(),
               max_tilts, tilt_steps, noises, orders, lambdas]

###################################################################################################
print('--Original distributions')

for key in mag_datas:
    mag_datas[key].quiver_plot3d(title='Original distribution ({})'.format(key))
    mlab.savefig('{}{}_analytic.png'.format(PATH, key), size=(800, 800))
    mlab.close(all=True)

###################################################################################################
print('--Reconstruction')

total_number = len(list(itertools.product(*config_list)))
print 'Number of configurations:', total_number

for i, configuration in enumerate(itertools.product(*config_list)):
    # Extract keys:
    mag_key, mask_key, xy_key, max_tilt, tilt_step, noise, order, lam = configuration
    print '----CONFIG {} of {}:'.format(i+1, total_number)
    print configuration
    name = '{}_{}_{}tilts_max{}_in{}steps_{}noise_{}order_{:.0E}lam'.format(*configuration)
    dim = mag_datas[mag_key].dim
    # Mask:
    if masks[mask_key]:
        mask = mag_datas[mag_key].get_mask()
    else:
        mask = np.ones(dim, dtype=bool)
    # DataSet:
    data = DataSet(a, dim, b_0, mask)
    # Tilts:
    tilts = np.arange(-max_tilt/180.*pi, max_tilt/180.*pi, tilt_step/180.*pi)
    # Projectors:
    projectors = []
    if xy_tilts[xy_key][0]:
        projectors.extend([XTiltProjector(dim, tilt) for tilt in tilts])
    if xy_tilts[xy_key][1]:
        projectors.extend([YTiltProjector(dim, tilt) for tilt in tilts])
    data.projectors = projectors
    # PhaseMaps:
    data.phase_maps = data.create_phase_maps(mag_datas[mag_key])
    # Noise:
    if noise != 0:
        for i, phase_map in enumerate(data.phase_maps):
            phase_map += PhaseMap(a, np.random.normal(0, noise, data.projectors[i].dim_uv))
            data.phase_maps[i] = phase_map
    # Regularisation:
    if order == 0:
        reg = ZeroOrderRegularisator(mask, lam, p=2)
    if order == 1:
        reg = FirstOrderRegularisator(mask, lam, p=2)
    # Reconstruction:
    info = []
    with TakeTime('reconstruction'):
        mag_data_rec = rc.optimize_linear(data, regularisator=reg, max_iter=100, info=info)
    # Plots:
    mag_data_rec.save_to_netcdf4('{}{}.nc'.format(PATH, name))
    mag_data_rec.quiver_plot3d('Reconstructed distribution ({})'.format(mag_key))
    mlab.savefig('{}{}_REC.png'.format(PATH, name), size=(800, 800))
    (mag_data_rec - mag_datas[mag_key]).quiver_plot3d('Difference ({})'.format(mag_key))
    mlab.savefig('{}{}_DIFF.png'.format(PATH, name), size=(800, 800))
    mlab.close(all=True)
    data_shelve = shelve.open(PATH+'3d_shelve')
    data_shelve[name] = info
    data_shelve.close()
    print 'chisq = {:.6f}, chisq_m = {:.6f}, chisq_a = {:.6f}'.format(*info)
    gc.collect()
    print 'RSS = {:.2f} MB'.format(proc.memory_info().rss/1024.**2)

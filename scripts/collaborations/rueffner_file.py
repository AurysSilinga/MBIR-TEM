# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 14:37:11 2014

@author: Jan
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from pylab import griddata

from tqdm import tqdm

import tables

import pyramid
from pyramid.magdata import MagData
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMConvolve

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
###################################################################################################
PATH = '../../output/'

#mag_file = netCDF4.Dataset(PATH+'dyn_0090mT_dyn.h5_dat.h5')
#
#print mag_file
#
#print mag_file.groups['data'].groups['fields']
#
#data = mag_file.groups['data']
#
#fields = data.groups['fields']


#points = netCDF4.Dataset(PATH+'dyn_0090mT_dyn.h5_dat.h5').groups['mesh'].variables['points'][...]

dim = (16, 182+40, 210+40)
a = 1.
b_0 = 1.1
projector_z = SimpleProjector(dim, axis='z')
projector_x = SimpleProjector(dim, axis='x')
pm_z = PMConvolve(a, projector_z, b_0)
pm_x = PMConvolve(a, projector_z, b_0)

h5file = tables.openFile(PATH+'dyn_0090mT_dyn.h5_dat.h5')
points = h5file.root.mesh.points.read()

axis = plt.figure().add_subplot(1, 1, 1, aspect='equal')
axis.scatter(points[:, 0], points[:, 1])
mlab.points3d(points[:, 0], points[:, 1], points[:, 2], mode='2dvertex')

#data_old = h5file.root.data.fields.m.read(field='m_CoFeb')[0, ...]

# Filling zeros:
iz_x = np.concatenate([np.linspace(-74, -37, 20),
                       np.linspace(-37, 37, 20),
                       np.linspace(37, 74, 20),
                       np.linspace(74, 37, 20),
                       np.linspace(37, -37, 20),
                       np.linspace(-37, -74, 20)])
iz_y = np.concatenate([np.linspace(0, 64, 20),
                       np.linspace(64, 64, 20),
                       np.linspace(64, 0, 20),
                       np.linspace(0, -64, 20),
                       np.linspace(-64, -64, 20),
                       np.linspace(-64, 0, 20)])
iz_z = np.zeros(len(iz_x))

# Set up grid:
xs, ys, zs = np.arange(-105, 105), np.arange(-91, 91), np.arange(-8, 8)
xx, yy = np.meshgrid(xs, ys)

#test = []

for t in np.arange(0, 1001, 5):
    print 't =', t
    vectors = h5file.root.data.fields.m.read(field='m_CoFeb')[t, ...]
    data = np.hstack((points, vectors))
#    if data_old is not None:
#        np.testing.assert_equal(data, data_old)
#    data_old = np.copy(data)

    zs_unique = np.unique(data[:, 2])  # TODO: not used

    # Create empty magnitude:
    magnitude = np.zeros((3, len(zs), len(ys), len(xs)))

    for i, z in tqdm(enumerate(zs), total=len(zs)):
    #    print z
    #    print a/2
    #    print np.abs(data[:, 2]-z)
        z_slice = data[np.abs(data[:, 2]-z) <= a/2., :]
        weights = 1 - np.abs(z_slice[:, 2]-z)*2/a
    #    print z_slice.shape, z
    #    print z_slice
    #    print weights
        for j in range(3):  # For all 3 components!
    #        if z <= 0:
            grid_x = np.concatenate([z_slice[:, 0], iz_x])
            grid_y = np.concatenate([z_slice[:, 1], iz_y])
            grid_z = np.concatenate([weights*z_slice[:, 3+j], iz_z])
    #        else:
    #        grid_x = z_slice[:, 0]
    #        grid_y = z_slice[:, 1]
    #        grid_z = weights*z_slice[:, 3+j]
            grid = griddata(grid_x, grid_y, grid_z, xx, yy)
            magnitude[j, i, :, :] = grid.filled(fill_value=0)

    mag_data = MagData(a, magnitude)
    mag_data.pad(20, 20, 0)
    phase_map = pm(mag_data)
    phase_map.unit = 'mrad'
    phase_map.display_combined(density=100, interpolation='bilinear', limit=None,
                               grad_encode='bright')[0]
    plt.savefig(PATH+'rueffner/phase_map_t_'+str(t)+'.png')
#    plt.close('all')
#    mag_data.quiver_plot()
#    mag_data.save_to_x3d('rueffner.x3d')
    mag_data.scale_down()
    mag_data.quiver_plot3d()

# -*- coding: utf-8 -*-
"""Created on Fri Jul 25 14:37:11 2014 @author: Jan"""

import os

import numpy as np
import matplotlib.pyplot as plt
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
dim = (16, 190, 220)
dim_uv = (300, 500)
a = 1.
b_0 = 1.1
dens_z = 4E3
dens_x = 1E2
lim_z = 22
lim_x = 0.35
###################################################################################################
# Build projectors and phasemapper:
projector_z = SimpleProjector(dim, axis='z', dim_uv=dim_uv)
projector_x = SimpleProjector(dim, axis='x', dim_uv=dim_uv)
pm_z = PMConvolve(a, projector_z, b_0)
pm_x = PMConvolve(a, projector_x, b_0)
# Read in hdf5-file and extract points:
h5file = tables.openFile(PATH+'dyn_0090mT_dyn.h5_dat.h5')
points = h5file.root.mesh.points.read()
# Plot point distribution in 2D and 3D:
axis = plt.figure().add_subplot(1, 1, 1, aspect='equal')
axis.scatter(points[:, 0], points[:, 1])
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
xs = np.arange(-dim[2]/2, dim[2]/2)
ys = np.arange(-dim[1]/2, dim[1]/2)
zs = np.arange(-dim[0]/2, dim[0]/2)
xx, yy = np.meshgrid(xs, ys)
# Interpolation and phase calculation for all timesteps:
for t in np.arange(865, 1001, 5):
    print 't =', t
    vectors = h5file.root.data.fields.m.read(field='m_CoFeb')[t, ...]
    data = np.hstack((points, vectors))
    # Create empty magnitude:
    magnitude = np.zeros((3, len(zs), len(ys), len(xs)))
    # Go over all z-slices:
    for i, z in tqdm(enumerate(zs), total=len(zs)):
        z_slice = data[np.abs(data[:, 2]-z) <= a/2., :]
        weights = 1 - np.abs(z_slice[:, 2]-z)*2/a
        for j in range(3):  # For all 3 components!
            grid_x = np.concatenate([z_slice[:, 0], iz_x])
            grid_y = np.concatenate([z_slice[:, 1], iz_y])
            grid_z = np.concatenate([weights*z_slice[:, 3+j], iz_z])
            grid = griddata(grid_x, grid_y, grid_z, xx, yy)
            magnitude[j, i, :, :] = grid.filled(fill_value=0)
    # Create magnetic distribution and phase maps:
    mag_data = MagData(a, magnitude)
    phase_map_z = pm_z(mag_data)
    phase_map_x = pm_x(mag_data)
    phase_map_z.unit = 'mrad'
    # Display phase maps and save them to png:
    phase_map_z.display_combined(density=dens_z, interpolation='bilinear', limit=lim_z)
    plt.savefig(PATH+'rueffner/phase_map_z_t_'+str(t)+'.png')
    phase_map_x.display_combined(density=dens_x, interpolation='bilinear', limit=lim_x)
    plt.savefig(PATH+'rueffner/phase_map_x_t_'+str(t)+'.png')
    # Close all plots to avoid clutter:
    plt.close('all')

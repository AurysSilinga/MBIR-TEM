# -*- coding: utf-8 -*-
"""Created on Fri Jan 24 11:17:11 2014 @author: Jan"""


import os

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib._pylab_helpers import Gcf
from pylab import griddata
from mayavi import mlab

import pickle
import vtk
from tqdm import tqdm

import pyramid
from pyramid.magdata import MagData
from pyramid.projector import SimpleProjector, YTiltProjector, XTiltProjector
from pyramid.phasemapper import PMAdapterFM, PMConvolve
from pyramid.dataset import DataSet

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
###################################################################################################
PATH = '../../output/vtk data/longtube_withcap/CoFeB_tube_cap_4nm'
b_0 = 1.54
gain = 1
force_calculation = False
###################################################################################################
# Load vtk-data:
if force_calculation or not os.path.exists(PATH+'.pickle'):
    # Setting up reader:
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(PATH+'.vtk')
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    # Getting output:
    output = reader.GetOutput()
    # Reading points and vectors:
    size = output.GetNumberOfPoints()
    vtk_points = output.GetPoints().GetData()
    vtk_vectors = output.GetPointData().GetVectors()
    # Converting points to numpy array:
    point_array = np.zeros(vtk_points.GetSize())
    vtk_points.ExportToVoidPointer(point_array)
    point_array = np.reshape(point_array, (-1, 3))
    # Converting vectors to numpy array:
    vector_array = np.zeros(vtk_points.GetSize())
    vtk_vectors.ExportToVoidPointer(vector_array)
    vector_array = np.reshape(vector_array, (-1, 3))
    # Combining data:
    data = np.hstack((point_array, vector_array))
    with open(PATH+'.pickle', 'w') as pf:
        pickle.dump(data, pf)
else:
    with open(PATH+'.pickle') as pf:
        data = pickle.load(pf)
# Scatter plot of all x-y-coordinates
axis = plt.figure().add_subplot(1, 1, 1, aspect='equal')
axis.scatter(data[data[:, 2] <= 0, 0], data[data[:, 2] <= 0, 1])
axis
mlab.points3d(data[:, 0], data[:, 1], data[:, 2], mode='2dvertex')
plt.show()

###################################################################################################
# Interpolate on regular grid:
if force_calculation or not os.path.exists(PATH+'.nc'):

    # Determine the size of object:
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    z_min, z_max = data[:, 2].min(), data[:, 2].max()
    x_diff, y_diff, z_diff = np.ptp(data[:, 0]), np.ptp(data[:, 1]), np.ptp(data[:, 2])
    x_cent, y_cent, z_cent = x_min+x_diff/2., y_min+y_diff/2., z_min+z_diff/2.
    # Filling zeros:
    iz_x = np.concatenate([np.linspace(-2.95, -2.95, 20),
                           np.linspace(-2.95, 0, 20),
                           np.linspace(0, 2.95, 20),
                           np.linspace(2.95, 2.95, 20),
                           np.linspace(2.95, 0, 20),
                           np.linspace(0, -2.95, 20), ])
    iz_y = np.concatenate([np.linspace(-1.70, 1.70, 20),
                           np.linspace(1.70, 3.45, 20),
                           np.linspace(3.45, 1.70, 20),
                           np.linspace(1.70, -1.70, 20),
                           np.linspace(-1.70, -3.45, 20),
                           np.linspace(-3.45, -1.70, 20), ])
    # Find unique z-slices:
    zs_unique = np.unique(data[:, 2])
    # Determine the grid spacing (in 1/10 nm):
    a = (zs_unique[1] - zs_unique[0])
    # Set actual z-slices:
    zs = np.arange(z_min, z_max, a)
    # Create regular grid:
    xs = np.arange(x_cent-x_diff, x_cent+x_diff, a)
    ys = np.arange(y_cent-y_diff, y_cent+y_diff, a)
    xx, yy = np.meshgrid(xs, ys)
    # Create empty magnitude:
    magnitude = np.zeros((3, len(zs), len(ys), len(xs)))

    for i, z in tqdm(enumerate(zs), total=len(zs)):
#        n = bisect.bisect(zs_unique, z)
#        z_lo, z_up = zs_unique[n], zs_unique[n+1]
        z_slice = data[np.abs(data[:, 2]-z) <= a/2., :]
        weights = 1 - np.abs(z_slice[:, 2]-z)*2/a
#        print n, z_slice.shape, z, z_lo, z_up
#        print z_slice
#        print weights
        for j in range(3):  # For all 3 components!
            if z <= 0:
                grid_x = np.concatenate([z_slice[:, 0], iz_x])
                grid_y = np.concatenate([z_slice[:, 1], iz_y])
                grid_z = np.concatenate([weights*z_slice[:, 3+j], np.zeros(len(iz_x))])
            else:
                grid_x = z_slice[:, 0]
                grid_y = z_slice[:, 1]
                grid_z = weights*z_slice[:, 3+j]
            grid = griddata(grid_x, grid_y, grid_z, xx, yy)
            magnitude[j, i, :, :] = grid.filled(fill_value=0)

    a = a*10  # convert to nm
#    print a

#    for i, z in tqdm(enumerate(zs), total=len(zs)):
#        n = bisect.bisect(zs_unique, z)
#        z_lo, z_up = zs_unique[n], zs_unique[n+1]
#        slice_lo = data[data[:, 2] == z_lo, :]
#        slice_up = data[data[:, 2] == z_up, :]
#        print n, slice_up.shape, z, z_lo, z_up
#        print slice_up
#        if slice_up.shape[0] < 3:
#            continue
#        for j in range(3):  # For all 3 components!
#            # Lower layer:
#            grid_lo = griddata(slice_lo[:, 0], slice_lo[:, 1], slice_lo[:, 3 + j], xx, yy)
#            # Upper layer:
#            grid_up = griddata(slice_up[:, 0], slice_up[:, 1], slice_up[:, 3 + j], xx, yy)
#            # Interpolated layer:
#            grid_interpol = (z-z_lo)/(z_up-z_lo)*grid_lo + (z_up-z)/(z_up-z_lo)*grid_up
#            magnitude[j, i

#    # WITH MASKING OF THE CENTER (SYMMETRIC):
#    iz_x = np.concatenate([np.linspace(-2.95, -2.95, 20),
#                           np.linspace(-2.95, 0, 20),
#                           np.linspace(0, 2.95, 20),
#                           np.linspace(2.95, 2.95, 20),
#                           np.linspace(2.95, 0, 20),
#                           np.linspace(0, -2.95, 20), ])
#    iz_y = np.concatenate([np.linspace(-1.70, 1.70, 20),
#                           np.linspace(1.70, 3.45, 20),
#                           np.linspace(3.45, 1.70, 20),
#                           np.linspace(1.70, -1.70, 20),
#                           np.linspace(-1.70, -3.45, 20),
#                           np.linspace(-3.45, -1.70, 20), ])
#    for i, z in tqdm(enumerate(zs), total=len(zs)):
#        subdata = data[data[:, 2] == z, :]
#        for j in range(3):  # For all 3 components!
#            gridded_subdata = griddata(np.concatenate([subdata[:, 0], iz_x]),
#                                       np.concatenate([subdata[:, 1], iz_y]),
#                                       np.concatenate([subdata[:, 3 + j], np.zeros(len(iz_x))]),
#                                       o, p)
#            magnitude[j, i, :, :] = gridded_subdata.filled(fill_value=0)

#    # WITH MASKING OF THE CENTER (ASYMMETRIC):
#    iz_x = np.concatenate([np.linspace(-5.88, -5.88, 50),
#                           np.linspace(-5.88, 0, 50),
#                            np.linspace(0, 5.88, 50),
#                            np.linspace(5.88, 5.88, 50),
#                            np.linspace(5.88, 0, 50),
#                            np.linspace(0, -5.88, 50),])
#    iz_y = np.concatenate([np.linspace(-2.10, 4.50, 50),
#                           np.linspace(4.50, 7.90, 50),
#                            np.linspace(7.90, 4.50, 50),
#                            np.linspace(4.50, -2.10, 50),
#                            np.linspace(-2.10, -5.50, 50),
#                            np.linspace(-5.50, -2.10, 50), ])
#    for i, z in tqdm(enumerate(zs), total=len(zs)):
#        subdata = data[data[:, 2] == z, :]
#        for j in range(3):  # For all 3 components!
#            gridded_subdata = griddata(np.concatenate([subdata[:, 0], iz_x]),
#                                       np.concatenate([subdata[:, 1], iz_y]),
#                                       np.concatenate([subdata[:, 3 + j], np.zeros(len(iz_x))]),
#                                       o, p)
#            magnitude[j, i, :, :] = gridded_subdata.filled(fill_value=0)

#    # WITHOUT MASKING OF THE CENTER:
#    for i, z in tqdm(enumerate(zs), total=len(zs)):
#        subdata = data[data[:, 2] == z, :]
#        print subdata.shape, z, zs_temp
#        if subdata.shape[0] < 3:
#            continue
#        for j in range(3):  # For all 3 components!
#            gridded_subdata = griddata(subdata[:, 0], subdata[:, 1], subdata[:, 3 + j], o, p)
#            magnitude[j, i, :, :] = gridded_subdata.filled(fill_value=0)

    # Creating MagData object:
    mag_data = MagData(a, np.pad(magnitude, ((0, 0), (0, 0), (0, 1), (0, 1)), mode='constant',
                                 constant_values=0))
    mag_data.save_to_netcdf4(PATH+'.nc')
else:
    mag_data = MagData.load_from_netcdf4(PATH+'.nc')
mag_data.quiver_plot(proj_axis='x')
###################################################################################################
# Phasemapping:
projector = SimpleProjector(mag_data.dim)
phasemapper = PMAdapterFM(mag_data.a, projector)
phase_map = phasemapper(mag_data)
(-phase_map).display_combined(title=r'Combined Plot (B$_0$={} T, Cos x {})'.format(b_0, gain),
                              density=gain)
plt.savefig(PATH+'_{}T_cosx{}.png'.format(b_0, gain))
(-phase_map).display_combined(title=r'Combined Plot (B$_0$={} T, Cos x {})'.format(b_0, gain),
                              density=gain, interpolation='bilinear')
plt.savefig(PATH+'_{}T_cosx{}_smooth.png'.format(b_0, gain))
mag_data.scale_down()
mag_data.save_to_netcdf4(PATH+'_scaled.nc')
mag_data_tip = MagData(mag_data.a, mag_data.magnitude[:, 350:, :, :])
mag_data_tip.save_to_netcdf4(PATH+'_tip_scaled.nc')
mag_data_tip.quiver_plot3d()

count = 16

dim_uv_x = (500, 100)
dim_uv_y = (100, 500)

density = 8

tilts_full = np.linspace(-pi/2, pi/2, num=count/2, endpoint=False)
tilts_miss = np.linspace(-pi/3, pi/3, num=count/2, endpoint=False)

projectors_y = [YiltProjector(mag_data.dim, tilt, dim_uv=dim_uv_y) for tilt in tilts_miss]
projectors_x = [XTiltProjector(mag_data.dim, tilt, dim_uv=dim_uv_x) for tilt in tilts_miss]
projectors = np.concatenate((projectors_y, projectors_x))
phasemappers = [PMConvolve(mag_data.a, proj, b_0) for proj in projectors]

data_set = DataSet(mag_data.a, dim_uv_x, b_0)

for i, pm in enumerate(phasemappers):
    data_set.append((pm(mag_data), projectors[i]))

plt.close('all')

data_set.display_combined(density=density, interpolation='bilinear')

figures = [manager.canvas.figure for manager in Gcf.get_all_fig_managers()]

for i, figure in enumerate(figures):
    figure.savefig(PATH+'_figure{}.png'.format(i))

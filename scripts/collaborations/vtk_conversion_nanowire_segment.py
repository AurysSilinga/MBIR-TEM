# -*- coding: utf-8 -*-
"""Created on Fri Jan 24 11:17:11 2014 @author: Jan"""


import os

import numpy as np
import matplotlib.pyplot as plt
from pylab import griddata

import pickle
import vtk
from tqdm import tqdm

import pyramid
from pyramid.magdata import MagData
from pyramid.phasemapper import pm

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
###################################################################################################
PATH = '../../output/vtk data/tube_90x30x50_sat_edge_equil.gmr'
b_0 = 1.54
gain = 12
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
axis = plt.figure().add_subplot(1, 1, 1)
axis.scatter(data[:, 0], data[:, 1])
plt.show()
###################################################################################################
# Interpolate on regular grid:
if force_calculation or not os.path.exists(PATH+'.nc'):
    # Find unique z-slices:
    zs = np.unique(data[:, 2])
    # Determine the grid spacing:
    a = zs[1] - zs[0]
    # Determine the size of object:
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    z_min, z_max = data[:, 2].min(), data[:, 2].max()
    x_diff, y_diff, z_diff = np.ptp(data[:, 0]), np.ptp(data[:, 1]), np.ptp(data[:, 2])
    x_cent, y_cent, z_cent = x_min+x_diff/2., y_min+y_diff/2., z_min+z_diff/2.
    # Create regular grid
    xs = np.arange(x_cent-x_diff, x_cent+x_diff, a)
    ys = np.arange(y_cent-y_diff, y_cent+y_diff, a)
    o, p = np.meshgrid(xs, ys)
    # Create empty magnitude:
    magnitude = np.zeros((3, len(zs), len(ys), len(xs)))

    # WITH MASKING OF THE CENTER (SYMMETRIC):
    iz_x = np.concatenate([np.linspace(-4.95, -4.95, 50),
                           np.linspace(-4.95, 0, 50),
                           np.linspace(0, 4.95, 50),
                           np.linspace(4.95, 4.95, 50),
                           np.linspace(-4.95, 0, 50),
                           np.linspace(0, 4.95, 50), ])
    iz_y = np.concatenate([np.linspace(-2.88, 2.88, 50),
                           np.linspace(2.88, 5.7, 50),
                           np.linspace(5.7, 2.88, 50),
                           np.linspace(2.88, -2.88, 50),
                           np.linspace(-2.88, -5.7, 50),
                           np.linspace(-5.7, -2.88, 50), ])
    for i, z in tqdm(enumerate(zs), total=len(zs)):
        subdata = data[data[:, 2] == z, :]
        for j in range(3):  # For all 3 components!
            gridded_subdata = griddata(np.concatenate([subdata[:, 0], iz_x]),
                                       np.concatenate([subdata[:, 1], iz_y]),
                                       np.concatenate([subdata[:, 3 + j], np.zeros(len(iz_x))]),
                                       o, p)
            magnitude[j, i, :, :] = gridded_subdata.filled(fill_value=0)

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
#        for j in range(3):  # For all 3 components!
#            gridded_subdata = griddata(subdata[:, 0], subdata[:, 1], subdata[:, 3 + j], o, p)
#            magnitude[j, i, :, :] = gridded_subdata.filled(fill_value=0)

    # Creating MagData object:
    mag_data = MagData(0.2*10, magnitude)
    mag_data.save_to_netcdf4(PATH+'.nc')
else:
    mag_data = MagData.load_from_netcdf4(PATH+'.nc')
mag_data.quiver_plot()
###################################################################################################
# Phasemapping:
phase_map = pm(mag_data, dim_uv=(300, 300))
(-phase_map).display_combined(title=r'Combined Plot (B$_0$={} T, Cos x {})'.format(b_0, gain),
                              gain=gain)
plt.savefig(PATH+'_{}T_cosx{}.png'.format(b_0, gain))
(-phase_map).display_combined(title=r'Combined Plot (B$_0$={} T, Cos x {})'.format(b_0, gain),
                              gain=gain, interpolation='bilinear')
plt.savefig(PATH+'_{}T_cosx{}_smooth.png'.format(b_0, gain))

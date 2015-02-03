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
from pyramid.projector import XTiltProjector
from pyramid.phasemapper import PhaseMapperRDFC
from pyramid.kernel import Kernel

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
###################################################################################################
PATH = '../../output/vtk data/tube_160x30x1100nm/02758'
b_0 = 1.54
gain = 8
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
# Turn magnetization around by 90° around x-axis:
#magnitude_new = np.zeros((3, mag_data.dim[1], mag_data.dim[0], mag_data.dim[2]))
#for i in range(mag_data.dim[2]):
#    x_rot = np.rot90(mag_data.magnitude[0, ..., i]).copy()
#    y_rot = np.rot90(mag_data.magnitude[1, ..., i]).copy()
#    z_rot = np.rot90(mag_data.magnitude[2, ..., i]).copy()
#    magnitude_new[0, ..., i] = x_rot
#    magnitude_new[1, ..., i] = z_rot
#    magnitude_new[2, ..., i] = -y_rot
#mag_data.magnitude = magnitude_new
#mag_data.save_to_netcdf4(PATH+'_lying_down.nc')


dim = mag_data.dim
dim_uv = (500, 200)
angles = [0, 20, 40, 60]

mag_data_xy = mag_data.copy()
mag_data_xy.magnitude[2] = 0

mag_data_z = mag_data.copy()
mag_data_z.magnitude[0] = 0
mag_data_z.magnitude[1] = 0

# Iterate over all angles:
for angle in angles:
    angle_rad = np.pi/2 + angle*np.pi/180
    projector = XTiltProjector(dim, angle_rad, dim_uv)
    mag_proj = projector(mag_data_z)
    phase_map = PhaseMapperRDFC(Kernel(mag_data.a, projector.dim_uv))(mag_proj)
    phase_map.display_combined('Phase Map Nanowire Tip', gain=gain,
                               interpolation='bilinear')
    plt.savefig(PATH+'_nanowire_z_xtilt_{}.png'.format(angle), dpi=500)
    mag_proj.scale_down(2)
    axis = mag_proj.quiver_plot()
    plt.savefig(PATH+'_nanowire_z_mag_xtilt_{}.png'.format(angle), dpi=500)
    axis = mag_proj.quiver_plot(log=True)
    plt.savefig(PATH+'_nanowire_z_mag_log_xtilt_{}.png'.format(angle), dpi=500)
    # Close plots:
    plt.close('all')


#mag_data.scale_down(2)
#mag_data.quiver_plot3d()
#
#mag_data_xy.scale_down(2)
#mag_data_xy.quiver_plot3d()
#
#mag_data_z.scale_down(2)
#mag_data_z.quiver_plot3d()

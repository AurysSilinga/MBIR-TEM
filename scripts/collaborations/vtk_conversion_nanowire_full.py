# -*- coding: utf-8 -*-
"""Created on Fri Jan 24 11:17:11 2014 @author: Jan"""


import os

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib._pylab_helpers import Gcf
from pylab import griddata

import pickle
import vtk
from tqdm import tqdm

import pyramid
from pyramid.magdata import MagData
from pyramid.projector import YTiltProjector, XTiltProjector
from pyramid.phasemapper import PMConvolve
from pyramid.phasemap import PhaseMap
from pyramid.dataset import DataSet

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
###################################################################################################
PATH = '../../output/vtk data/longtube_withcap/CoFeB_tube_cap_4nm'
b_0 = 1.54
force_calculation = False
count = 16
dim_uv_x = (500, 100)
dim_uv_y = (100, 500)
density = 8
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
###################################################################################################
# Interpolate on regular grid:
if force_calculation or not os.path.exists(PATH+'.nc'):
    # Determine the size of the object:
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
    # Go over all slices:
    for i, z in tqdm(enumerate(zs), total=len(zs)):
        z_slice = data[np.abs(data[:, 2]-z) <= a/2., :]
        weights = 1 - np.abs(z_slice[:, 2]-z)*2/a
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
    # Creating MagData object and save it as netcdf-file:
    mag_data = MagData(a, np.pad(magnitude, ((0, 0), (0, 0), (0, 1), (0, 1)), mode='constant',
                                 constant_values=0))
    mag_data.save_to_netcdf4(PATH+'.nc')
else:
    mag_data = MagData.load_from_netcdf4(PATH+'.nc')
###################################################################################################
mag_data_scaled = mag_data.copy()
mag_data_scaled.scale_down()
mag_data_scaled.save_to_netcdf4(PATH+'_scaled.nc')
# Create tilts and projectors:
tilts_full = np.linspace(-pi/2, pi/2, num=count/2, endpoint=False)
tilts_miss = np.linspace(-pi/3, pi/3, num=count/2, endpoint=False)
projectors_y = [YTiltProjector(mag_data_scaled.dim, tilt, dim_uv=dim_uv_y) for tilt in tilts_miss]
projectors_x = [XTiltProjector(mag_data_scaled.dim, tilt, dim_uv=dim_uv_x) for tilt in tilts_miss]
pm_y = [PMConvolve(mag_data_scaled.a, proj, b_0) for proj in projectors_y]
pm_x = [PMConvolve(mag_data_scaled.a, proj, b_0) for proj in projectors_x]
# Create data sets for x and y tilts:
data_set_y = DataSet(mag_data_scaled.a, dim_uv_y, b_0)
data_set_x = DataSet(mag_data_scaled.a, dim_uv_x, b_0)
# Create and append phase maps and projectors:
for i in range(len(pm_y)):
    data_set_y.append((pm_y[i](mag_data_scaled), projectors_y[i]))
    data_set_x.append((pm_x[i](mag_data_scaled), projectors_x[i]))
# Display phase maps:
data_set_y.display_combined(density=density, interpolation='bilinear')
data_set_x.display_combined(density=density, interpolation='bilinear')
# Save figures:
figures = [manager.canvas.figure for manager in Gcf.get_all_fig_managers()]
for i, figure in enumerate(figures):
    figure.savefig(PATH+'_figure{}.png'.format(i))
plt.close('all')
###################################################################################################
# Close ups:
dim = (300, 72, 72)
dim_uv = (600, 150)
angles = [0, 20, 40, 60, -20, -40, -60]
for angle in angles:
    angle_rad = angle * np.pi/180
    shift = int(np.abs(80*np.sin(angle_rad)))
    projector = XTiltProjector(dim, np.pi/2+angle_rad, dim_uv)
    projector_scaled = XTiltProjector((dim[0]/2, dim[1]/2, dim[2]/2), np.pi/2+angle_rad,
                                      (dim_uv[0]/2, dim_uv[1]/2))
    # Tip:
    mag_data_tip = MagData(mag_data.a, mag_data.magnitude[:, 608:, ...])
    pm = PMConvolve(mag_data.a, projector)
    phase_map_tip = PhaseMap(mag_data.a, pm(mag_data_tip).phase[350-shift:530-shift, :])
    phase_map_tip.display_combined('Phase Map Nanowire Tip', density=density,
                                   interpolation='bilinear')
    plt.savefig(PATH+'_nanowire_tip_xtilt_{}.png'.format(angle))
    mag_data_tip.scale_down()
    mag_proj_tip = projector_scaled.to_mag_data(mag_data_tip)
    axis = mag_proj_tip.quiver_plot()
    axis.set_xlim(17, 55)
    axis.set_ylim(180-shift/2, 240-shift/2)
    plt.savefig(PATH+'_nanowire_tip_mag_xtilt_{}.png'.format(angle))
    axis = mag_proj_tip.quiver_plot(log=True)
    axis.set_xlim(17, 55)
    axis.set_ylim(180-shift/2, 240-shift/2)
    plt.savefig(PATH+'_nanowire_tip_mag_log_xtilt_{}.png'.format(angle))
    # Bottom:
    mag_data_bot = MagData(mag_data.a, mag_data.magnitude[:, :300, ...])
    pm = PMConvolve(mag_data.a, projector)
    phase_map_tip = PhaseMap(mag_data.a, pm(mag_data_bot).phase[50+shift:225+shift, :])
    phase_map_tip.display_combined('Phase Map Nanowire Bottom', density=density,
                                   interpolation='bilinear')
    plt.savefig(PATH+'_nanowire_bot_xtilt_{}.png'.format(angle))
    mag_data_bot.scale_down()
    mag_proj_bot = projector_scaled.to_mag_data(mag_data_bot)
    axis = mag_proj_bot.quiver_plot()
    axis.set_xlim(17, 55)
    axis.set_ylim(50+shift/2, 110+shift/2)
    plt.savefig(PATH+'_nanowire_bot_mag_xtilt_{}.png'.format(angle))
    axis = mag_proj_bot.quiver_plot(log=True)
    axis.set_xlim(17, 55)
    axis.set_ylim(50+shift/2, 110+shift/2)
    plt.savefig(PATH+'_nanowire_bot_mag_log_xtilt_{}.png'.format(angle))
    # Close plots:
    plt.close('all')

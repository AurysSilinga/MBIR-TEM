# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:15:08 2014

@author: Jan
"""


import numpy as np
from pyramid.magdata import MagData
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMAdapterFM, PMFourier
from matplotlib.ticker import FuncFormatter

data = np.loadtxt('../output/data from Edinburgh/long_grain_remapped_0p0035.txt', delimiter=',')

a = 1000 * (data[1, 2] - data[0, 2])

dim = len(np.unique(data[:, 2])), len(np.unique(data[:, 1])), len(np.unique(data[:, 0]))

mag_vec = np.concatenate([data[:, 3], data[:, 4], data[:, 5]])

x_mag = np.reshape(data[:, 3], dim, order='F')
y_mag = np.reshape(data[:, 4], dim, order='F')
z_mag = np.reshape(data[:, 5], dim, order='F')

magnitude = np.array((x_mag, y_mag, z_mag))

mag_data = MagData(a, magnitude)

#mag_data.pad(30, 20, 0)
#
#mag_data.scale_up()
#
#mag_data.quiver_plot()

#mag_data.quiver_plot3d()

projector = SimpleProjector(mag_data.dim)

phasemapper = PMAdapterFM(mag_data.a, projector)
phasemapper = PMFourier(mag_data.a, projector, padding = 1)

phase_map = phasemapper(mag_data)

phase_axis = phase_map.display_combined(density=20, interpolation='bilinear', grad_encode='bright')[0]

phase_axis.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:3.0f}'.format(x*mag_data.a)))
phase_axis.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:3.0f}'.format(x*mag_data.a)))

#phase_map.display_phase3d()

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:06:01 2014

@author: Jan
"""

import numpy as np
from numpy import pi

from pyramid.magdata import MagData
from pyramid.projector import YTiltProjector
from pyramid.phasemapper import PMConvolve

from time import clock

import matplotlib.pyplot as plt

#data = np.loadtxt('../output/data from Edinburgh/long_grain_remapped_0p0035.txt', delimiter=',')
#
#a = 1000 * (data[1, 2] - data[0, 2])
#
#dim = len(np.unique(data[:, 2])), len(np.unique(data[:, 1])), len(np.unique(data[:, 0]))
#
#mag_vec = np.concatenate([data[:, 3], data[:, 4], data[:, 5]])
#
#x_mag = np.reshape(data[:, 3], dim, order='F')
#y_mag = np.reshape(data[:, 4], dim, order='F')
#z_mag = np.reshape(data[:, 5], dim, order='F')
#
#magnitude = np.array((x_mag, y_mag, z_mag))
#
#mag_data = MagData(a, magnitude)
#
#mag_data.pad(30, 20, 0)


PATH = '../output/vtk data/tube_90x30x50_sat_edge_equil.gmr'

mag_data = MagData.load_from_netcdf4(PATH+'.nc')

#for tilt in np.linspace(0, 2*pi, num=20, endpoint=False):
#    projector = YTiltProjector(mag_data.dim, tilt)
#    phasemapper = PMConvolve(mag_data.a, projector)
#    phase_map = phasemapper(mag_data)
#    (-phase_map).display_combined(title=u'Tilt series $(\phi = {:2.1f} \pi)$'.format(tilt/pi),
#                                  limit=2., density=12, interpolation='bilinear', 
#                                  grad_encode='bright')
#    plt.savefig(PATH+'_tilt_{:3.2f}pi.png'.format(tilt/pi))


mag_data.scale_down()

mag_data.quiver_plot3d()

#projectors = [YTiltProjector(mag_data.dim, i)
#              for i in np.linspace(0, 2*pi, num=20, endpoint=False)]
#
#start = clock()
#phasemapper = PMConvolve[PMConvolve(mag_data.a, projector) for projector in projectors]
#print 'Overhead  :', clock()-start
#
#start = clock()
#phase_maps = [pm(mag_data) for pm in phasemappers]
#print 'Phasemapping:', clock()-start
#
#[phase_map.display_combined(density=12, interpolation='bilinear', grad_encode='bright')
# for phase_map in phase_maps]

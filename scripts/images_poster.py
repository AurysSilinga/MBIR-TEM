# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 19:19:19 2014

@author: Jan
"""

import os

from numpy import pi
import numpy as np

import pyramid.magcreator as mc
import pyramid.analytic as an
from pyramid.magdata import MagData
from pyramid.phasemapper import PMConvolve
from pyramid.projector import SimpleProjector
from pyramid.phasemap import PhaseMap

import matplotlib.pyplot as plt

from time import clock

#directory = '../output/poster'
#if not os.path.exists(directory):
#    os.makedirs(directory)
## Input parameters:
#a = 10.0  # nm
#dim = (64, 128, 128)
## Slab:f
#center = (32, 32, 32)  # in px (z, y, x), index starts with 0!
#width = (322, 48, 48)  # in px (z, y, x)
#mag_shape_slab = mc.Shapes.slab(dim, center, width)
## Disc:
#center = (32, 32, 96)  # in px (z, y, x), index starts with 0!
#radius = 24  # in px
#height = 24  # in px
#mag_shape_disc = mc.Shapes.disc(dim, center, radius, height)
## Sphere:
#center = (32, 96, 64)  # in px (z, y, x), index starts with 0!
#radius = 24  # in px
#mag_shape_sphere = mc.Shapes.sphere(dim, center, radius)
## Create empty MagData object and add magnetized objects:
#mag_data = MagData(a, mc.create_mag_dist_homog(mag_shape_slab, pi/6))
#mag_data += MagData(a, mc.create_mag_dist_vortex(mag_shape_disc, (32, 96)))
#mag_data += MagData(a, mc.create_mag_dist_homog(mag_shape_sphere, -pi/4))
## Plot the magnetic distribution, phase map and holographic contour map:
#mag_data_coarse = mag_data.copy()
#mag_data_coarse.scale_down(2)
#mag_data_coarse.quiver_plot()
#plt.savefig(os.path.join(directory, 'mag.png'))
#mag_data_coarse.quiver_plot3d()
#projector = SimpleProjector(dim)
#phase_map = PMConvolve(a, projector, b_0=0.1)(mag_data)
#phase_map.display_phase()
#plt.savefig(os.path.join(directory, 'phase.png'))
#phase_map.display_holo(density=4, interpolation='bilinear')
#plt.savefig(os.path.join(directory, 'holo.png'))
#
#
#dim = (1, 9, 9)
#mag_shape = mc.Shapes.pixel(dim, (int(dim[0]/2), int(dim[1]/2), int(dim[2]/2)))
#mag_data_x = MagData(a, mc.create_mag_dist_homog(mag_shape, 0))
#mag_data_y = MagData(a, mc.create_mag_dist_homog(mag_shape, pi/2))
#phasemapper = PMConvolve(a, SimpleProjector(dim))
#phasemapper(mag_data_x).display_phase()
#phasemapper(mag_data_y).display_phase()


a = 1.
dim = (128, 128, 128)
center = (dim[0]/2, dim[1]/2, dim[2]/2)
radius = dim[1]/4  # in px
mag_shape = mc.Shapes.sphere(dim, center, radius)
mag_data = MagData(a, mc.create_mag_dist_homog(mag_shape, pi/4))
phasemapper = PMConvolve(a, SimpleProjector(dim))
start = clock()
phase_map = phasemapper(mag_data)
print 'TIME:', clock() - start
phase_ana = an.phase_mag_sphere(dim, a, pi/4, center, radius)
phase_diff = (phase_ana - phase_map).phase
print 'RMS%:', np.sqrt(np.mean((phase_diff)**2))/phase_ana.phase.max()*100
print 'max:', phase_ana.phase.max()
print 'RMS:', np.sqrt(np.mean((phase_diff)**2))
PhaseMap(a, phase_diff).display_phase()

mag_data.scale_down(2)
mag_data.quiver_plot()
mag_data.quiver_plot3d()
phase_map.display_phase()


# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 19:19:19 2014

@author: Jan
"""

import os

from numpy import pi
import numpy as np

import pyramid
import pyramid.magcreator as mc
import pyramid.analytic as an
from pyramid.magdata import MagData
from pyramid.phasemapper import pm, PhaseMapperRDFC
from pyramid.projector import SimpleProjector
from pyramid.kernel import Kernel
from pyramid.phasemap import PhaseMap

from time import clock

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)


a = 10.0  # nm
dim = (1, 9, 9)
mag_shape = mc.Shapes.pixel(dim, (int(dim[0]/2), int(dim[1]/2), int(dim[2]/2)))
mag_data_x = MagData(a, mc.create_mag_dist_homog(mag_shape, 0))
mag_data_y = MagData(a, mc.create_mag_dist_homog(mag_shape, pi/2))
pm(mag_data_x).display_phase()
pm(mag_data_y).display_phase()


a = 1
dim = (128, 128, 128)
center = (dim[0]/2, dim[1]/2, dim[2]/2)
radius = dim[1]/4  # in px
mag_shape = mc.Shapes.sphere(dim, center, radius)
mag_data = MagData(a, mc.create_mag_dist_homog(mag_shape, pi/4))
projector = SimpleProjector(dim)
phasemapper = PhaseMapperRDFC(Kernel(a, projector.dim_uv))
start = clock()
phase_map = phasemapper(projector(mag_data))
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

# -*- coding: utf-8 -*-
"""Created on Fri Feb 28 14:25:59 2014 @author: Jan"""


import os

import numpy as np
from numpy import pi

from time import clock

import pyramid
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap
from pyramid.projector import YTiltProjector, XTiltProjector
from pyramid.phasemapper import PMConvolve
from pyramid.dataset import DataSet
import pyramid.magcreator as mc

from pyramid.kernel import Kernel
from pyramid.forwardmodel import ForwardModel
from pyramid.costfunction import Costfunction
import pyramid.reconstruction as rc

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)

###################################################################################################
print('--Generating input phase_maps')

a = 10.
b_0 = 1.
dim = (8, 32, 32)
dim_uv = dim[1:3]
count = 8

#center = (int(dim[0]/2)-0.5, int(dim[1]/2)-0.5, int(dim[2]/2)-0.5)
#mag_shape = mc.Shapes.disc(dim, center, dim[2]/4, dim[2]/2)
#magnitude = mc.create_mag_dist_vortex(mag_shape)
#mag_data = MagData(a, magnitude)
#mag_data.quiver_plot3d()

vortex_shape = mc.Shapes.disc(dim, (3.5, 9.5, 9.5), 5, 4)
sphere_shape = mc.Shapes.sphere(dim, (3.5, 22.5, 9.5), 3)
slab_shape = mc.Shapes.slab(dim, (3.5, 15.5, 22.5), (5, 8, 8))
mag_data = MagData(a, mc.create_mag_dist_vortex(vortex_shape, (3.5, 9.5, 9.5)))
mag_data += MagData(a, mc.create_mag_dist_homog(sphere_shape, pi/4, pi/4))
mag_data += MagData(a, mc.create_mag_dist_homog(slab_shape, -pi/6))

mag_data.quiver_plot3d()

tilts_full = np.linspace(-pi/2, pi/2, num=count/2, endpoint=False)
tilts_miss = np.linspace(-pi/3, pi/3, num=count/2, endpoint=False)

projectors_xy_full, projectors_x_full, projectors_y_full = [], [], []
projectors_xy_miss, projectors_x_miss, projectors_y_miss = [], [], []
projectors_x_full.extend([XTiltProjector(dim, tilt) for tilt in tilts_full])
projectors_y_full.extend([YTiltProjector(dim, tilt) for tilt in tilts_full])
projectors_xy_full.extend(projectors_x_full)
projectors_xy_full.extend(projectors_y_full)
projectors_x_miss.extend([XTiltProjector(dim, tilt) for tilt in tilts_miss])
projectors_y_miss.extend([YTiltProjector(dim, tilt) for tilt in tilts_miss])
projectors_xy_miss.extend(projectors_x_miss)
projectors_xy_miss.extend(projectors_y_miss)

###################################################################################################
projectors = projectors_xy_full
noise = 0.0
###################################################################################################

phasemappers = [PMConvolve(mag_data.a, projector, b_0) for projector in projectors]
if noise != 0:
    phase_maps = [pm(mag_data) + PhaseMap(a, np.random.normal(0, noise, dim_uv))
                  for pm in phasemappers]
else:
    phase_maps = [pm(mag_data) for pm in phasemappers]

###################################################################################################
print('--Setting up data collection')

dim_uv = dim[1:3]

lam = 10**-10

data = DataSet(a, dim_uv, b_0)
[data.append((phase_maps[i], projectors[i])) for i in range(len(projectors))]
y = data.phase_vec
kern = Kernel(data.a, data.dim_uv, data.b_0)
F = ForwardModel(data.projectors, kern)
C = Costfunction(y, F, lam)

data.display()

###################################################################################################
#print('--Test simple solver')
#
#start = clock()
#mag_data_opt = rc.optimize_sparse_cg(data, verbosity=1)
#print 'Time:', str(clock()-start)
#mag_data_opt.quiver_plot3d()
#(mag_data_opt - mag_data).quiver_plot3d()
##data.display(data.create_phase_maps(mag_data_opt))

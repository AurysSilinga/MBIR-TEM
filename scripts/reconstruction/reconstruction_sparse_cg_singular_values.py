# -*- coding: utf-8 -*-
"""Created on Thu Apr 24 11:00:00 2014 @author: Jan"""


import os

import numpy as np
from numpy import pi

import pyramid
from pyramid.magdata import MagData
from pyramid.projector import YTiltProjector, XTiltProjector
from pyramid.dataset import DataSet
from pyramid.phasemap import PhaseMap
from pyramid.forwardmodel import ForwardModel

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)

###################################################################################################
print('--Singular value decomposition')

a = 1.
b_0 = 1.
dim = (3, 3, 3)
dim_uv = dim[1:3]
count = 8

tilts_full = np.linspace(-pi/2, pi/2, num=count/2, endpoint=False)
tilts_miss = np.linspace(-pi/3, pi/3, num=count/2, endpoint=False)

phase_zero = PhaseMap(a, np.zeros(dim_uv))

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
###################################################################################################

size_2d = np.prod(dim_uv)
size_3d = np.prod(dim)

data = DataSet(a, dim, b_0)
[data.append(phase_zero, projectors[i]) for i in range(len(projectors))]

y = data.phase_vec
F = ForwardModel(data)

M = np.asmatrix([F.jac_dot(None, np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
# MTM = M.T * M + lam * np.asmatrix(np.eye(3*size_3d))

U, s, V = np.linalg.svd(M)  # TODO: M or MTM?

for i in range(len(s)):
    print 'Singular value:', s[i]
    title = 'Singular value = {:g}'.format(s[i])
    MagData(data.a, np.array(V[i, :]).reshape((3,)+dim)).quiver_plot3d(title)

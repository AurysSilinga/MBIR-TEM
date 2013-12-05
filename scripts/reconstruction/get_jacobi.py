# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 11:15:38 2013

@author: Jan
"""

import time
import os

import numpy as np
from numpy import pi

import pyramid.magcreator as mc
import pyramid.projector as pj
import pyramid.phasemapper as pm
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap


directory = '../../output/reconstruction'
if not os.path.exists(directory):
    os.makedirs(directory)
filename = directory + '/jacobi.npy'
b_0 = 1.0  # in T
dim = (1, 64, 64)  # in px (y,x)
res = 10.0  # in nm
phi = pi/4

center = (0, int(dim[1]/2), int(dim[2]/2))  # in px (y,x) index starts with 0!
width = (0, 1, 1)  # in px (y,x)

mag_data = MagData(res, mc.create_mag_dist_homog(mc.Shapes.slab(dim, center, width), phi))
projection = pj.simple_axis_projection(mag_data)
print 'Compare times for multiplication of a vector (ones) with the jacobi matrix!'

# Prepare kernel and vector:
dim_proj = np.shape(projection[0])
size = np.prod(dim_proj)
kernel = pm.Kernel(dim_proj, res, 'disc')
vector = np.ones(2*size)

# Calculation with kernel (core):
tic = time.clock()
result_core = kernel.multiply_jacobi_core(vector)
toc = time.clock()
print 'Calculation with kernel (core):   ', toc - tic

# Calculation with kernel (core2):
tic = time.clock()
result_core2 = kernel.multiply_jacobi_core2(vector)
toc = time.clock()
print 'Calculation with kernel (core2):  ', toc - tic

# Calculation with kernel:
tic = time.clock()
result_kernel = kernel.multiply_jacobi(vector)
toc = time.clock()
print 'Calculation with kernel:          ', toc - tic

# Calculation during phasemapping:
tic = time.clock()
jacobi = np.zeros((size, 2*size))
PhaseMap(res, pm.phase_mag_real(res, projection, b_0, 'disc', jacobi=jacobi))
result_during = np.asarray(np.asmatrix(jacobi)*np.asmatrix(vector).T).reshape(-1)
toc = time.clock()
print 'Calculation during phasemapping:  ', toc - tic

# Test if calculations match:
np.testing.assert_almost_equal(result_kernel, result_during)
np.testing.assert_almost_equal(result_core, result_during)
np.testing.assert_almost_equal(result_core2, result_during)
print 'Calculations give the same result!'

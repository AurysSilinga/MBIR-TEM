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
dim = (1, 8, 8)  # in px (y,x)
res = 10.0  # in nm
phi = pi/4

center = (0, int(dim[1]/2), int(dim[2]/2))  # in px (y,x) index starts with 0!
width = (0, 1, 1)  # in px (y,x)

mag_data = MagData(res, mc.create_mag_dist_homog(mc.Shapes.slab(dim, center, width), phi))
projection = pj.simple_axis_projection(mag_data)
print 'Projection calculated!'

'''NUMERICAL SOLUTION'''
# numerical solution Real Space:
dim_proj = np.shape(projection[0])
size = np.prod(dim_proj)
kernel = pm.Kernel(dim_proj, res, 'disc')

tic = time.clock()
kernel.multiply_jacobi(np.ones(2*size))  # column-wise
toc = time.clock()
print 'Time for one multiplication with the Jacobi-Matrix:      ', toc - tic

jacobi = np.zeros((size, 2*size))
tic = time.clock()
phase_map = PhaseMap(res, pm.phase_mag_real(res, projection, b_0, 'disc', jacobi=jacobi))
toc = time.clock()
phase_map.display()
np.savetxt(filename, jacobi)
print 'Time for Jacobi-Matrix during phase calculation:         ', toc - tic

tic = time.clock()
jacobi_test = kernel.get_jacobi()
toc = time.clock()
print 'Time for Jacobi-Matrix from the Kernel:                  ', toc - tic

unity = np.eye(2*size)
jacobi_test2 = np.zeros((size, 2*size))
tic = time.clock()
for i in range(unity.shape[1]):
    jacobi_test2[:, i] = kernel.multiply_jacobi(unity[:, i])  # column-wise
toc = time.clock()
print 'Time for getting the Jacobi-Matrix (vector-wise):        ', toc - tic

unity_transp = np.eye(size)
jacobi_transp = np.zeros((2*size, size))
tic = time.clock()
for i in range(unity_transp.shape[1]):
    jacobi_transp[:, i] = kernel.multiply_jacobi_T(unity_transp[:, i])  # column-wise
toc = time.clock()
print 'Time for getting the transp. Jacobi-Matrix (vector-wise):', toc - tic

print 'Methods (during vs kernel) in accordance?                ', \
    np.logical_not(np.all(jacobi-jacobi_test))
print 'Methods (during vs vector-wise) in accordance?           ', \
    np.logical_not(np.all(jacobi-jacobi_test2))
print 'Methods (transponed Jacobi) in accordance?               ', \
    np.logical_not(np.all(jacobi.T-jacobi_transp))

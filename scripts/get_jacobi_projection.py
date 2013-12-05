# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:45:13 2013

@author: Jan
"""

import pyramid.magcreator as mc
from pyramid.magdata import MagData
from pyramid.projector import Projection
from pyramid.kernel import Kernel

import numpy as np
from numpy import pi

import time



a = 1.0
dim = (3, 3, 3)
px = (1, 1, 1)
tilt = pi/3

mag_data = MagData(a, mc.create_mag_dist_homog(mc.Shapes.pixel(dim, px), phi=pi/4, theta=pi/4))

proj = Projection.single_tilt_projection(mag_data, tilt)

size_2d = dim[1] * dim[2]
size_3d = dim[0] * dim[1] * dim[2]

ref_u = proj.u
ref_v = proj.v

z_mag_vec = np.asmatrix(mag_data.magnitude[0].reshape(-1)).T
y_mag_vec = np.asmatrix(mag_data.magnitude[1].reshape(-1)).T
x_mag_vec = np.asmatrix(mag_data.magnitude[2].reshape(-1)).T

mag_vec = np.concatenate((x_mag_vec, y_mag_vec, z_mag_vec))



# Via full multiplicatoin with weight-matrix:

start = time.clock()

weight_matrix = np.asmatrix(proj.get_weight_matrix())

test_u_wf = (np.cos(tilt) * np.asarray(weight_matrix * x_mag_vec).reshape(dim[1], dim[2])
           + np.sin(tilt) * np.asarray(weight_matrix * z_mag_vec).reshape(dim[1], dim[2]))

test_v_wf = np.asarray(weight_matrix * y_mag_vec).reshape(dim[1], dim[2])

print 'Time for calculation via full weight-matrix multiplication:  ', time.clock() - start

np.testing.assert_almost_equal(test_u_wf, ref_u, err_msg='u-part does not match!')
np.testing.assert_almost_equal(test_v_wf, ref_v, err_msg='v-part does not match!')



# Via direct multiplication with weight_matrix:

start = time.clock()

test_u_wd = (np.cos(tilt) * proj.multiply_weight_matrix(x_mag_vec).reshape(dim[1], dim[2])
           + np.sin(tilt) * proj.multiply_weight_matrix(z_mag_vec).reshape(dim[1], dim[2]))

test_v_wd = proj.multiply_weight_matrix(y_mag_vec).reshape(dim[1], dim[2])

print 'Time for calculation via direct weight-matrix multiplication:', time.clock() - start

np.testing.assert_almost_equal(test_u_wd, ref_u, err_msg='u-part does not match!')
np.testing.assert_almost_equal(test_v_wd, ref_v, err_msg='v-part does not match!')



# Via full multiplication with jacobi-matrix:

start = time.clock()

jacobi = np.asmatrix(proj.get_jacobi())
projected_mag = np.asarray(jacobi * mag_vec).reshape(2, dim[1], dim[2])

test_u_jf = projected_mag[0, ...]
test_v_jf = projected_mag[1, ...]

print 'Time for calculation via full jacobi-matrix multiplication:  ', time.clock() - start

np.testing.assert_almost_equal(test_u_jf, ref_u, err_msg='u-part does not match!')
np.testing.assert_almost_equal(test_v_jf, ref_v, err_msg='v-part does not match!')



# Via direct multiplication with jacobi-matrix:

start = time.clock()

projected_mag = proj.multiply_jacobi(mag_vec).reshape(2, dim[1], dim[2])

test_u_jd = projected_mag[0, ...]
test_v_jd = projected_mag[1, ...]

print 'Time for calculation via direct jacobi-matrix multiplication:', time.clock() - start

np.testing.assert_almost_equal(test_u_jd, ref_u, err_msg='u-part does not match!')
np.testing.assert_almost_equal(test_v_jd, ref_v, err_msg='v-part does not match!')



# Via full multiplication with transposed jacobi-matrix:

start = time.clock()

jacobi_T = np.asmatrix(proj.get_jacobi()).T

test_T_jf = np.asarray(jacobi_T * np.asmatrix(np.ones(2*size_2d)).T).reshape(-1)

print 'Time for calculation via full transposed multiplication:     ', time.clock() - start




# Via direct multiplication with transposed jacobi-matrix:

start = time.clock()

jacobi_T = np.asmatrix(proj.get_jacobi()).T

test_complete_T = np.asarray(jacobi_T * np.asmatrix(np.ones(2*size_2d)).T).reshape(-1)

test_T_jd = proj.multiply_jacobi_T(np.ones(2*size_2d))

print 'Time for calculation via direct transposed multiplication:   ', time.clock() - start

np.testing.assert_almost_equal(test_T_jd, test_T_jf, err_msg='Transposed vector does not match!')



## Cost function testing:
#
#kern = Kernel((dim[1], dim[2]), a)
#
#identity = np.eye(5)
#
#right = kern.multiply_jacobi(proj.multiply_jacobi(mag_vec))
#cost = kern.multiply_jacobi_T(proj.multiply_jacobi_T(right))
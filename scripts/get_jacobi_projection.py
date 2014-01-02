# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:45:13 2013

@author: Jan
"""

import pyramid.magcreator as mc
from pyramid.magdata import MagData
from pyramid.projector import Projection, Projector
import pyramid.projector as pj
from pyramid.kernel import Kernel

import numpy as np
from numpy import pi

import time



a = 1.0
dim = (64, 64, 64)
px = (0, 0, 0)
tilt = pi/2

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

start = time.clock()
projector = Projector.single_tilt_z_projector(dim, tilt)
print 'Create Projector:', time.clock() - start

start = time.clock()
test_u, test_v = projector.mag_projection(mag_data)
print 'Projection (mag):', time.clock() - start

start = time.clock()
test_thickness = projector.thickness_projection(mag_data)
print 'Projection (thi):', time.clock() - start

start = time.clock()
x_ref_v, x_ref_u, _ = pj.simple_axis_projection(mag_data, axis='x')
print 'Projection (x ref):', time.clock() - start
start = time.clock()
y_ref_v, y_ref_u, _ = pj.simple_axis_projection(mag_data, axis='y')
print 'Projection (y ref):', time.clock() - start
start = time.clock()
z_ref_v, z_ref_u, _ = pj.simple_axis_projection(mag_data, axis='z')
print 'Projection (z ref):', time.clock() - start
start = time.clock()
x_test_u, x_test_v = Projector.main_axis_projector(dim, axis='x').mag_projection(mag_data)
print 'Projection (x new):', time.clock() - start
start = time.clock()
y_test_u, y_test_v = Projector.main_axis_projector(dim, axis='y').mag_projection(mag_data)
print 'Projection (y new):', time.clock() - start
start = time.clock()
z_test_u, z_test_v = Projector.main_axis_projector(dim, axis='z').mag_projection(mag_data)
print 'Projection (z new):', time.clock() - start

np.testing.assert_almost_equal(x_test_v, x_ref_v, err_msg='x: v-part does not match!')
np.testing.assert_almost_equal(x_test_u, x_ref_u, err_msg='x: u-part does not match!')
np.testing.assert_almost_equal(y_test_v, y_ref_v, err_msg='y: v-part does not match!')
np.testing.assert_almost_equal(y_test_u, y_ref_u, err_msg='y: u-part does not match!')
np.testing.assert_almost_equal(z_test_v, z_ref_v, err_msg='z: v-part does not match!')
np.testing.assert_almost_equal(z_test_u, z_ref_u, err_msg='z: u-part does not match!')


## Via full multiplicatoin with weight-matrix:
#
#start = time.clock()
#
#weight_matrix = np.asmatrix(proj.get_weight_matrix())
#
#test_u_wf = (np.cos(tilt) * np.asarray(weight_matrix * x_mag_vec).reshape(dim[1], dim[2])
#           + np.sin(tilt) * np.asarray(weight_matrix * z_mag_vec).reshape(dim[1], dim[2]))
#
#test_v_wf = np.asarray(weight_matrix * y_mag_vec).reshape(dim[1], dim[2])
#
#print 'Time for calculation via full weight-matrix multiplication:  ', time.clock() - start
#
#np.testing.assert_almost_equal(test_u_wf, ref_u, err_msg='u-part does not match!')
#np.testing.assert_almost_equal(test_v_wf, ref_v, err_msg='v-part does not match!')
#
#
#
## Via direct multiplication with weight_matrix:
#
#start = time.clock()
#
#test_u_wd = (np.cos(tilt) * proj.multiply_weight_matrix(x_mag_vec).reshape(dim[1], dim[2])
#           + np.sin(tilt) * proj.multiply_weight_matrix(z_mag_vec).reshape(dim[1], dim[2]))
#
#test_v_wd = proj.multiply_weight_matrix(y_mag_vec).reshape(dim[1], dim[2])
#
#print 'Time for calculation via direct weight-matrix multiplication:', time.clock() - start
#
#np.testing.assert_almost_equal(test_u_wd, ref_u, err_msg='u-part does not match!')
#np.testing.assert_almost_equal(test_v_wd, ref_v, err_msg='v-part does not match!')
#
#
#
## Via full multiplication with jacobi-matrix:
#
#start = time.clock()
#
#jacobi = np.asmatrix(proj.get_jacobi())
#projected_mag = np.asarray(jacobi * mag_vec).reshape(2, dim[1], dim[2])
#
#test_u_jf = projected_mag[0, ...]
#test_v_jf = projected_mag[1, ...]
#
#print 'Time for calculation via full jacobi-matrix multiplication:  ', time.clock() - start
#
#np.testing.assert_almost_equal(test_u_jf, ref_u, err_msg='u-part does not match!')
#np.testing.assert_almost_equal(test_v_jf, ref_v, err_msg='v-part does not match!')
#
#
#
## Via direct multiplication with jacobi-matrix:
#
#start = time.clock()
#
#projected_mag = proj.multiply_jacobi(mag_vec).reshape(2, dim[1], dim[2])
#
#test_u_jd = projected_mag[0, ...]
#test_v_jd = projected_mag[1, ...]
#
#print 'Time for calculation via direct jacobi-matrix multiplication:', time.clock() - start
#
#np.testing.assert_almost_equal(test_u_jd, ref_u, err_msg='u-part does not match!')
#np.testing.assert_almost_equal(test_v_jd, ref_v, err_msg='v-part does not match!')
#
#
#
## Via full multiplication with transposed jacobi-matrix:
#
#start = time.clock()
#
#jacobi_T = np.asmatrix(proj.get_jacobi()).T
#
#test_T_jf = np.asarray(jacobi_T * np.asmatrix(np.ones(2*size_2d)).T).reshape(-1)
#
#print 'Time for calculation via full transposed multiplication:     ', time.clock() - start
#
#
#
#
## Via direct multiplication with transposed jacobi-matrix:
#
#start = time.clock()
#
#jacobi_T = np.asmatrix(proj.get_jacobi()).T
#
#test_complete_T = np.asarray(jacobi_T * np.asmatrix(np.ones(2*size_2d)).T).reshape(-1)
#
#test_T_jd = proj.multiply_jacobi_T(np.ones(2*size_2d))
#
#print 'Time for calculation via direct transposed multiplication:   ', time.clock() - start
#
#np.testing.assert_almost_equal(test_T_jd, test_T_jf, err_msg='Transposed vector does not match!')
#
#
#
## Cost function testing:
#
#kern = Kernel((dim[1], dim[2]), a)
#
#identity = np.eye(5)
#
#right = kern.multiply_jacobi(proj.multiply_jacobi(mag_vec))
#left = proj.multiply_jacobi_T(kern.multiply_jacobi_T(np.asmatrix(right).T))


#start = time.clock()
#
#karl = Projection.single_tilt_projection_sparse(mag_data, tilt=tilt)
#
#print 'Time:', time.clock() - start
#
#start = time.clock()
#
#korl = Projection.single_tilt_projection(mag_data, tilt=tilt)
#
#print 'Time:', time.clock() - start
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:25:59 2014

@author: Jan
"""


import numpy as np
from numpy import pi

from pyramid.magdata import MagData
from pyramid.projector import YTiltProjector
from pyramid.phasemapper import PMConvolve
from pyramid.datacollection import DataCollection
import pyramid.optimizer as opt

from pyramid.kernel import Kernel
from pyramid.forwardmodel import ForwardModel
from pyramid.costfunction import Costfunction


a = 1.
b_0 = 1000.
dim = (3, 3, 3)
count = 1

###################################################################################################
print('--Generating input phase_maps')


magnitude = np.zeros((3,)+dim)
magnitude[:, int(dim[0]/2), int(dim[1]/2), int(dim[2]/2)] = 1.

mag_data = MagData(a, magnitude)
mag_data.quiver_plot3d()

tilts = np.linspace(0, 2*pi, num=count, endpoint=False)
projectors = [YTiltProjector(mag_data.dim, tilt) for tilt in tilts]
phasemappers = [PMConvolve(mag_data.a, projector, b_0) for projector in projectors]
phase_maps = [pm(mag_data) for pm in phasemappers]

#[phase_map.display_phase(title=u'Tilt series $(\phi = {:2.1f} \pi)$'.format(tilts[i]/pi))
#                         for i, phase_map in enumerate(phase_maps)]
phase_maps[0].display_phase()

###################################################################################################
print('--Setting up data collection')

dim_uv = dim[1:3]

data_collection = DataCollection(a, dim_uv, b_0)

[data_collection.append((phase_maps[i], projectors[i])) for i in range(count)]

###################################################################################################
print('--Test optimizer')

first_guess = MagData(a, np.zeros((3,)+dim))

first_guess.magnitude[1, int(dim[0]/2), int(dim[1]/2), int(dim[2]/2)] = 1
#first_guess.magnitude[0, int(dim[0]/2), int(dim[1]/2), int(dim[2]/2)] = -1

first_guess.quiver_plot3d()

phase_guess = PMConvolve(first_guess.a, projectors[0], b_0)(first_guess)

phase_guess.display_phase()

#mag_opt = opt.optimize_cg(data_collection, first_guess)
#
#mag_opt.quiver_plot3d()
#
#phase_opt = PMConvolve(mag_opt.a, projectors[0], b_0)(mag_opt)
#
#phase_opt.display_phase()


###################################################################################################
print('--Further testing')

data = data_collection
mag_0 = first_guess
x_0 = first_guess.mag_vec
y = data.phase_vec
kern = Kernel(data.a, data.dim_uv)
F = ForwardModel(data.projectors, kern, data.b_0)
C = Costfunction(y, F)

size_3d = np.prod(dim)

cost = C(first_guess.mag_vec)
cost_grad = C.jac(first_guess.mag_vec)
cost_grad_del_x = cost_grad.reshape((3, 3, 3, 3))[0, ...]
cost_grad_del_y = cost_grad.reshape((3, 3, 3, 3))[1, ...]
cost_grad_del_z = cost_grad.reshape((3, 3, 3, 3))[2, ...]




x_t = np.asmatrix(mag_data.mag_vec).T
y = np.asmatrix(y).T
K = np.array([F.jac_dot(x_0, np.eye(81)[:, i]) for i in range(81)]).T
K = np.asmatrix(K)
lam = 10. ** -10
KTK = K.T * K + lam * np.asmatrix(np.eye(81))
print lam, 
#print pylab.cond(KTK),
x_f = KTK.I * K.T * y
print (np.asarray(K * x_f - y) ** 2).sum(),
print (np.asarray(K * x_t - y) ** 2).sum()
print x_f
x_rec = np.asarray(x_f).reshape(3,3,3,3)
print x_rec[0, ...]
#KTK = K.T * K
#u,s,v = pylab.svd(KTK, full_matrices=1)
#si = np.zeros_like(s)
#
#si[s>lam] = 1. / s[s>lam]
#si=np.asmatrix(np.diag(si))
#KTKI = np.asmatrix(v).T * si * np.asmatrix(u)
#
#
#x_f = KTKI * K.T * y
#print x_f
#
#


###################################################################################################
print('--Compliance test')


# test F

# test F.jac

# test F.jac_dot

#F_evaluate = F(mag_data.mag_vec)
#F_jacobi = F.jac_dot(None, mag_data.mag_vec)
#np.testing.assert_equal(F_evaluate, F_jacobi)
#
#F_reverse = F.jac_T_dot(None, F_jacobi)
#np.testing.assert_equal(F_reverse, mag_data.mag_vec)



from scipy.sparse.linalg import cg

K = np.asmatrix([F.jac_dot(x_0, np.eye(81)[:, i]) for i in range(81)]).T
lam = 10. ** -10
KTK = K.T * K + lam * np.asmatrix(np.eye(81))

A = KTK#np.array([F(np.eye(81)[:, i]) for i in range(81)])

b = F.jac_T_dot(None, y)

b_test = np.asarray((K.T * y).T)[0]

x_f = cg(A, b_test)[0]

mag_data_rec = MagData(a, np.zeros((3,)+dim))

mag_data_rec.mag_vec = x_f

mag_data_rec.quiver_plot3d()





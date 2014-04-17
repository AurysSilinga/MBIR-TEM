# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:25:59 2014

@author: Jan
"""


import numpy as np
from numpy import pi

from pyramid.magdata import MagData
from pyramid.projector import YTiltProjector, XTiltProjector
from pyramid.phasemapper import PMConvolve
from pyramid.dataset import DataSet
import pyramid.magcreator as mc

from pyramid.kernel import Kernel
from pyramid.forwardmodel import ForwardModel
from pyramid.costfunction import Costfunction
import pyramid.reconstruction as rc

from scipy.sparse.linalg import cg, LinearOperator


###################################################################################################
print('--Compliance test for one projection')

a = 1.
b_0 = 1000.
dim = (2, 2, 2)
count = 1

magnitude = np.zeros((3,)+dim)
magnitude[:, int(dim[0]/2), int(dim[1]/2), int(dim[2]/2)] = 1.

mag_data = MagData(a, magnitude)

tilts = np.linspace(0, 2*pi, num=count, endpoint=False)
projectors = [YTiltProjector(mag_data.dim, tilt) for tilt in tilts]
phasemappers = [PMConvolve(mag_data.a, projector, b_0) for projector in projectors]
phase_maps = [pm(mag_data) for pm in phasemappers]

dim_uv = dim[1:3]

data_collection = DataSet(a, dim_uv, b_0)

[data_collection.append((phase_maps[i], projectors[i])) for i in range(count)]

data = data_collection
y = data.phase_vec
kern = Kernel(data.a, data.dim_uv, data.b_0)
F = ForwardModel(data.projectors, kern)
C = Costfunction(y, F)

size_2d = np.prod(dim[1]*dim[2])
size_3d = np.prod(dim)

P = np.array([data.projectors[0](np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
K = np.array([kern(np.eye(2*size_2d)[:, i]) for i in range(2*size_2d)]).T
F_mult = K.dot(P)
F_direct = np.array([F(np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
np.testing.assert_almost_equal(F_direct, F_mult)

P_jac = np.array([data.projectors[0].jac_dot(np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
K_jac = np.array([kern.jac_dot(np.eye(2*size_2d)[:, i]) for i in range(2*size_2d)]).T
F_jac_mult = K.dot(P)
F_jac_direct = np.array([F.jac_dot(None, np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
np.testing.assert_almost_equal(F_jac_direct, F_jac_mult)
np.testing.assert_almost_equal(F_direct, F_jac_direct)

P_t = np.array([data.projectors[0].jac_T_dot(np.eye(2*size_2d)[:, i]) for i in range(2*size_2d)]).T
K_t = np.array([kern.jac_T_dot(np.eye(size_2d)[:, i]) for i in range(size_2d)]).T
F_t_mult = P_t.dot(K_t)
F_t_direct = np.array([F.jac_T_dot(None, np.eye(size_2d)[:, i]) for i in range(size_2d)]).T

P_t_ref = P.T
K_t_ref = K.T
F_t_ref = F_mult.T
np.testing.assert_almost_equal(P_t, P_t_ref)
np.testing.assert_almost_equal(K_t, K_t_ref)
np.testing.assert_almost_equal(F_t_mult, F_t_ref)
np.testing.assert_almost_equal(F_t_direct, F_t_ref)

###################################################################################################
print('--Compliance test for two projections')

a = 1.
b_0 = 1000.
dim = (2, 2, 2)
count = 2

magnitude = np.zeros((3,)+dim)
magnitude[:, int(dim[0]/2), int(dim[1]/2), int(dim[2]/2)] = 1.

mag_data = MagData(a, magnitude)

tilts = np.linspace(0, 2*pi, num=count, endpoint=False)
projectors = [YTiltProjector(mag_data.dim, tilt) for tilt in tilts]
phasemappers = [PMConvolve(mag_data.a, projector, b_0) for projector in projectors]
phase_maps = [pm(mag_data) for pm in phasemappers]

dim_uv = dim[1:3]

data_collection = DataSet(a, dim_uv, b_0)

[data_collection.append((phase_maps[i], projectors[i])) for i in range(count)]

data = data_collection
y = data.phase_vec
kern = Kernel(data.a, data.dim_uv, data.b_0)
F = ForwardModel(data.projectors, kern)
C = Costfunction(y, F)

size_2d = np.prod(dim[1]*dim[2])
size_3d = np.prod(dim)

P0 = np.array([data.projectors[0](np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
P1 = np.array([data.projectors[1](np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
P = np.vstack((P0, P1))
K = np.array([kern(np.eye(2*size_2d)[:, i]) for i in range(2*size_2d)]).T
F_mult0 = K.dot(P0)
F_mult1 = K.dot(P1)
F_mult = np.vstack((F_mult0, F_mult1))
F_direct = np.array([F(np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
np.testing.assert_almost_equal(F_direct, F_mult)

P_jac0 = np.array([data.projectors[0].jac_dot(np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
P_jac1 = np.array([data.projectors[1].jac_dot(np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
P = np.vstack((P0, P1))
K_jac = np.array([kern.jac_dot(np.eye(2*size_2d)[:, i]) for i in range(2*size_2d)]).T
F_jac_mult0 = K.dot(P_jac0)
F_jac_mult1 = K.dot(P_jac1)
F_jac_mult = np.vstack((F_jac_mult0, F_jac_mult1))
F_jac_direct = np.array([F.jac_dot(None, np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
np.testing.assert_almost_equal(F_jac_direct, F_jac_mult)
np.testing.assert_almost_equal(F_direct, F_jac_direct)

P_t0 = np.array([data.projectors[0].jac_T_dot(np.eye(2*size_2d)[:, i]) for i in range(2*size_2d)]).T
P_t1 = np.array([data.projectors[1].jac_T_dot(np.eye(2*size_2d)[:, i]) for i in range(2*size_2d)]).T
P_t = np.hstack((P_t0, P_t1))
K_t = np.array([kern.jac_T_dot(np.eye(size_2d)[:, i]) for i in range(size_2d)]).T
F_t_mult0 = P_t0.dot(K_t)
F_t_mult1 = P_t1.dot(K_t)
F_t_mult = np.hstack((F_t_mult0, F_t_mult1))
F_t_direct = np.array([F.jac_T_dot(None, np.eye(count*size_2d)[:, i]) for i in range(count*size_2d)]).T

P_t_ref = P.T
K_t_ref = K.T
F_t_ref = F_mult.T
np.testing.assert_almost_equal(P_t, P_t_ref)
np.testing.assert_almost_equal(K_t, K_t_ref)
np.testing.assert_almost_equal(F_t_mult, F_t_ref)
np.testing.assert_almost_equal(F_t_direct, F_t_ref)








###################################################################################################
print('--STARTING RECONSTRUCTION')



###################################################################################################
print('--Generating input phase_maps')

a = 10.
b_0 = 1000.
dim = (8, 8, 8)
count = 32

magnitude = np.zeros((3,)+dim)
magnitude[:, 3:6, 3:6, 3:6] = 1# int(dim[0]/2), int(dim[1]/2), int(dim[2]/2)] = 1.
magnitude = mc.create_mag_dist_vortex(mc.Shapes.disc(dim, (3.5, 3.5, 3.5), 3, 4))

mag_data = MagData(a, magnitude)
mag_data.quiver_plot3d()

tilts = np.linspace(0, 2*pi, num=count/2, endpoint=False)
projectors = []
projectors.extend([XTiltProjector(mag_data.dim, tilt) for tilt in tilts])
projectors.extend([YTiltProjector(mag_data.dim, tilt) for tilt in tilts])
phasemappers = [PMConvolve(mag_data.a, projector, b_0) for projector in projectors]
phase_maps = [pm(mag_data) for pm in phasemappers]

#[phase_map.display_phase(title=u'Tilt series $(\phi = {:2.1f} \pi)$'.format(tilts[i%(count/2)]/pi))
#                         for i, phase_map in enumerate(phase_maps)]

###################################################################################################
print('--Setting up data collection')

dim_uv = dim[1:3]

lam =  10. ** -10

size_2d = np.prod(dim_uv)
size_3d = np.prod(dim)


data_collection = DataSet(a, dim_uv, b_0)

[data_collection.append((phase_maps[i], projectors[i])) for i in range(count)]

data = data_collection
y = data.phase_vec
kern = Kernel(data.a, data.dim_uv, data.b_0)
F = ForwardModel(data.projectors, kern)
C = Costfunction(y, F, lam)

###################################################################################################
print('--Test simple solver')

#M = np.asmatrix([F.jac_dot(None, np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
#MTM = M.T * M + lam * np.asmatrix(np.eye(3*size_3d))
#A = MTM#np.array([F(np.eye(81)[:, i]) for i in range(81)])

#class A_adapt(LinearOperator):
#
#    def __init__(self, FwdModel, lam, shape):
#        self.fwd = FwdModel
#        self.lam = lam
#        self.shape = shape
#
#    def matvec(self, vector):
#        return self.fwd.jac_T_dot(None, self.fwd.jac_dot(None, vector)) + self.lam*vector
#
#    @property
#    def shape(self):
#        return self.shape
#
#    @property
#    def dtype(self):
#        return np.dtype("d") # None #np.ones(1).dtype
#
## TODO: .shape in F und C
#
#b = F.jac_T_dot(None, y)
#
#A_fast = A_adapt(F, lam, (3*size_3d, 3*size_3d))
#
#i = 0
#def printit(_):
#    global i
#    i += 1
#    print i
#
#x_f, info = cg(A_fast, b, callback=printit)
#
#mag_data_rec = MagData(a, np.zeros((3,)+dim))
#
#mag_data_rec.mag_vec = x_f
#
#mag_data_rec.quiver_plot3d()

#phase_maps_rec = [pm(mag_data_rec) for pm in phasemappers]
#[phase_map.display_phase(title=u'Tilt series (rec.) $(\phi = {:2.1f} \pi)$'.format(tilts[i%(count/2)]/pi))
#                         for i, phase_map in enumerate(phase_maps_rec)]

mag_data_opt = rc.optimize_sparse_cg(data_collection)

mag_data_opt.quiver_plot3d()


#first_guess = MagData(a, np.zeros((3,)+dim))
#
#first_guess.magnitude[1, int(dim[0]/2), int(dim[1]/2), int(dim[2]/2)] = 1
#first_guess.magnitude[0, int(dim[0]/2), int(dim[1]/2), int(dim[2]/2)] = -1
#
#first_guess.quiver_plot3d()
#
#phase_guess = PMConvolve(first_guess.a, projectors[0], b_0)(first_guess)
#
#phase_guess.display_phase()

#mag_opt = opt.optimize_cg(data_collection, first_guess)
#
#mag_opt.quiver_plot3d()
#
#phase_opt = PMConvolve(mag_opt.a, projectors[0], b_0)(mag_opt)
#
#phase_opt.display_phase()



###################################################################################################
print('--Singular value decomposition')

#a = 1.
#b_0 = 1000.
#dim = (3, 3, 3)
#count = 8
#
#magnitude = np.zeros((3,)+dim)
#magnitude[:, int(dim[0]/2), int(dim[1]/2), int(dim[2]/2)] = 1.
#
#mag_data = MagData(a, magnitude)
#mag_data.quiver_plot3d()
#
#tilts = np.linspace(0, 2*pi, num=count/2, endpoint=False)
#projectors = []
#projectors.extend([XTiltProjector(mag_data.dim, tilt) for tilt in tilts])
#projectors.extend([YTiltProjector(mag_data.dim, tilt) for tilt in tilts])
#phasemappers = [PMConvolve(mag_data.a, projector, b_0) for projector in projectors]
#phase_maps = [pm(mag_data) for pm in phasemappers]
#
##[phase_map.display_phase(title=u'Tilt series $(\phi = {:2.1f} \pi)$'.format(tilts[i%(count/2)]/pi))
##                         for i, phase_map in enumerate(phase_maps)]
#
#dim_uv = dim[1:3]
#
#lam =  10. ** -10
#
#size_2d = np.prod(dim_uv)
#size_3d = np.prod(dim)
#
#
#data_collection = DataCollection(a, dim_uv, b_0)
#
#[data_collection.append((phase_maps[i], projectors[i])) for i in range(count)]
#
#data = data_collection
#y = data.phase_vec
#kern = Kernel(data.a, data.dim_uv, data.b_0)
#F = ForwardModel(data.projectors, kern)
#C = Costfunction(y, F, lam)
#
#mag_data_opt = opt.optimize_sparse_cg(data_collection)
#mag_data_opt.quiver_plot3d()



#M = np.asmatrix([F.jac_dot(None, np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
#MTM = M.T * M + lam * np.asmatrix(np.eye(3*size_3d))
#A = MTM#np.array([F(np.eye(81)[:, i]) for i in range(81)])
#
#
#
#U, s, V = np.linalg.svd(M)
##np.testing.assert_almost_equal(U.T, V, decimal=5)
#
#for value in range(20):
#    print 'Singular value:', s[value]
#    MagData(data.a, np.array(V[value,:]).reshape((3,)+dim)).quiver_plot3d()
#
#
#for value in range(-10,0):
#    print 'Singular value:', s[value]
#    MagData(data.a, np.array(V[value,:]).reshape((3,)+dim)).quiver_plot3d()


# TODO: print all singular vectors for a 2x2x2 distribution for each single and both tilt series!

# TODO: Separate the script for SVD and compliance tests

####################################################################################################
#print('--Further testing')
#
#data = data_collection
#mag_0 = first_guess
#x_0 = first_guess.mag_vec
#y = data.phase_vec
#kern = Kernel(data.a, data.dim_uv)
#F = ForwardModel(data.projectors, kern, data.b_0)
#C = Costfunction(y, F)
#
#size_3d = np.prod(dim)
#
#cost = C(first_guess.mag_vec)
#cost_grad = C.jac(first_guess.mag_vec)
#cost_grad_del_x = cost_grad.reshape((3, 3, 3, 3))[0, ...]
#cost_grad_del_y = cost_grad.reshape((3, 3, 3, 3))[1, ...]
#cost_grad_del_z = cost_grad.reshape((3, 3, 3, 3))[2, ...]
#
#
#
#
#x_t = np.asmatrix(mag_data.mag_vec).T
#y = np.asmatrix(y).T
#K = np.array([F.jac_dot(x_0, np.eye(81)[:, i]) for i in range(81)]).T
#K = np.asmatrix(K)
#lam = 10. ** -10
#KTK = K.T * K + lam * np.asmatrix(np.eye(81))
#print lam,
##print pylab.cond(KTK),
#x_f = KTK.I * K.T * y
#print (np.asarray(K * x_f - y) ** 2).sum(),
#print (np.asarray(K * x_t - y) ** 2).sum()
#print x_f
#x_rec = np.asarray(x_f).reshape(3,3,3,3)
#print x_rec[0, ...]
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











#K = np.asmatrix([F.jac_dot(None, np.eye(24)[:, i]) for i in range(24)]).T
#lam = 10. ** -10
#KTK = K.T * K + lam * np.asmatrix(np.eye(24))
#
#A = KTK#np.array([F(np.eye(81)[:, i]) for i in range(81)])
#
#b = F.jac_T_dot(None, y)
#
#b_test = np.asarray((K.T.dot(y)).T)[0]
#
#x_f = cg(A, b_test)[0]
#
#mag_data_rec = MagData(a, np.zeros((3,)+dim))
#
#mag_data_rec.mag_vec = x_f
#
#mag_data_rec.quiver_plot3d()





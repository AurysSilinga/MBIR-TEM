# -*- coding: utf-8 -*-
"""Testcase for the reconstructor module."""


import os
import unittest


class TestCaseReconstruction(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_phasemapper')

    def tearDown(self):
        self.path = None

    def test_reconstruct_simple_leastsq(self):
        raise AssertionError


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseReconstruction)
    unittest.TextTestRunner(verbosity=2).run(suite)


###################################################################################################
#print('--Compliance test for one projection')
#
#a = 1.
#b_0 = 1000.
#dim = (2, 2, 2)
#count = 1
#
#magnitude = np.zeros((3,)+dim)
#magnitude[:, int(dim[0]/2), int(dim[1]/2), int(dim[2]/2)] = 1.
#
#mag_data = MagData(a, magnitude)
#
#tilts = np.linspace(0, 2*pi, num=count, endpoint=False)
#projectors = [YTiltProjector(mag_data.dim, tilt) for tilt in tilts]
#phasemappers = [PMConvolve(mag_data.a, projector, b_0) for projector in projectors]
#phase_maps = [pm(mag_data) for pm in phasemappers]
#
#dim_uv = dim[1:3]
#
#data_collection = DataSet(a, dim_uv, b_0)
#
#[data_collection.append((phase_maps[i], projectors[i])) for i in range(count)]
#
#data = data_collection
#y = data.phase_vec
#kern = Kernel(data.a, data.dim_uv, data.b_0)
#F = ForwardModel(data.projectors, kern)
#C = Costfunction(y, F)
#
#size_2d = np.prod(dim[1]*dim[2])
#size_3d = np.prod(dim)
#
#P = np.array([data.projectors[0](np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
#K = np.array([kern(np.eye(2*size_2d)[:, i]) for i in range(2*size_2d)]).T
#F_mult = K.dot(P)
#F_direct = np.array([F(np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
#np.testing.assert_almost_equal(F_direct, F_mult)
#
#P_jac = np.array([data.projectors[0].jac_dot(np.eye(3*size_3d)[:, i])
#                  for i in range(3*size_3d)]).T
#K_jac = np.array([kern.jac_dot(np.eye(2*size_2d)[:, i]) for i in range(2*size_2d)]).T
#F_jac_mult = K.dot(P)
#F_jac_direct = np.array([F.jac_dot(None, np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
#np.testing.assert_almost_equal(F_jac_direct, F_jac_mult)
#np.testing.assert_almost_equal(F_direct, F_jac_direct)
#
#P_t = np.array([data.projectors[0].jac_T_dot(np.eye(2*size_2d)[:, i])
#                for i in range(2*size_2d)]).T
#K_t = np.array([kern.jac_T_dot(np.eye(size_2d)[:, i]) for i in range(size_2d)]).T
#F_t_mult = P_t.dot(K_t)
#F_t_direct = np.array([F.jac_T_dot(None, np.eye(size_2d)[:, i]) for i in range(size_2d)]).T
#
#P_t_ref = P.T
#K_t_ref = K.T
#F_t_ref = F_mult.T
#np.testing.assert_almost_equal(P_t, P_t_ref)
#np.testing.assert_almost_equal(K_t, K_t_ref)
#np.testing.assert_almost_equal(F_t_mult, F_t_ref)
#np.testing.assert_almost_equal(F_t_direct, F_t_ref)
#
###################################################################################################
#print('--Compliance test for two projections')
#
#a = 1.
#b_0 = 1000.
#dim = (2, 2, 2)
#count = 2
#
#magnitude = np.zeros((3,)+dim)
#magnitude[:, int(dim[0]/2), int(dim[1]/2), int(dim[2]/2)] = 1.
#
#mag_data = MagData(a, magnitude)
#
#tilts = np.linspace(0, 2*pi, num=count, endpoint=False)
#projectors = [YTiltProjector(mag_data.dim, tilt) for tilt in tilts]
#phasemappers = [PMConvolve(mag_data.a, projector, b_0) for projector in projectors]
#phase_maps = [pm(mag_data) for pm in phasemappers]
#
#dim_uv = dim[1:3]
#
#data_collection = DataSet(a, dim_uv, b_0)
#
#[data_collection.append((phase_maps[i], projectors[i])) for i in range(count)]
#
#data = data_collection
#y = data.phase_vec
#kern = Kernel(data.a, data.dim_uv, data.b_0)
#F = ForwardModel(data.projectors, kern)
#C = Costfunction(y, F)
#
#size_2d = np.prod(dim[1]*dim[2])
#size_3d = np.prod(dim)
#
#P0 = np.array([data.projectors[0](np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
#P1 = np.array([data.projectors[1](np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
#P = np.vstack((P0, P1))
#K = np.array([kern(np.eye(2*size_2d)[:, i]) for i in range(2*size_2d)]).T
#F_mult0 = K.dot(P0)
#F_mult1 = K.dot(P1)
#F_mult = np.vstack((F_mult0, F_mult1))
#F_direct = np.array([F(np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
#np.testing.assert_almost_equal(F_direct, F_mult)
#
#P_jac0 = np.array([data.projectors[0].jac_dot(np.eye(3*size_3d)[:, i])
#                  for i in range(3*size_3d)]).T
#P_jac1 = np.array([data.projectors[1].jac_dot(np.eye(3*size_3d)[:, i])
#                  for i in range(3*size_3d)]).T
#P = np.vstack((P0, P1))
#K_jac = np.array([kern.jac_dot(np.eye(2*size_2d)[:, i]) for i in range(2*size_2d)]).T
#F_jac_mult0 = K.dot(P_jac0)
#F_jac_mult1 = K.dot(P_jac1)
#F_jac_mult = np.vstack((F_jac_mult0, F_jac_mult1))
#F_jac_direct = np.array([F.jac_dot(None, np.eye(3*size_3d)[:, i]) for i in range(3*size_3d)]).T
#np.testing.assert_almost_equal(F_jac_direct, F_jac_mult)
#np.testing.assert_almost_equal(F_direct, F_jac_direct)
#
#P_t0 = np.array([data.projectors[0].jac_T_dot(np.eye(2*size_2d)[:, i])
#                for i in range(2*size_2d)]).T
#P_t1 = np.array([data.projectors[1].jac_T_dot(np.eye(2*size_2d)[:, i])
#                for i in range(2*size_2d)]).T
#P_t = np.hstack((P_t0, P_t1))
#K_t = np.array([kern.jac_T_dot(np.eye(size_2d)[:, i]) for i in range(size_2d)]).T
#F_t_mult0 = P_t0.dot(K_t)
#F_t_mult1 = P_t1.dot(K_t)
#F_t_mult = np.hstack((F_t_mult0, F_t_mult1))
#F_t_direct = np.array([F.jac_T_dot(None, np.eye(count*size_2d)[:, i])
#                      for i in range(count*size_2d)]).T
#
#P_t_ref = P.T
#K_t_ref = K.T
#F_t_ref = F_mult.T
#np.testing.assert_almost_equal(P_t, P_t_ref)
#np.testing.assert_almost_equal(K_t, K_t_ref)
#np.testing.assert_almost_equal(F_t_mult, F_t_ref)
#np.testing.assert_almost_equal(F_t_direct, F_t_ref)

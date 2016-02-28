# -*- coding: utf-8 -*-
"""Testcase for the costfunction module"""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyramid.costfunction import Costfunction
from pyramid.dataset import DataSet
from pyramid.forwardmodel import ForwardModel
from pyramid.phasemap import PhaseMap
from pyramid.projector import SimpleProjector
from pyramid.regularisator import FirstOrderRegularisator


class TestCaseCostfunction(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_costfunction')
        self.a = 10.
        self.dim = (4, 5, 6)
        self.mask = np.zeros(self.dim, dtype=bool)
        self.mask[1:-1, 1:-1, 1:-1] = True
        self.data = DataSet(self.a, self.dim, mask=self.mask)
        self.projector = SimpleProjector(self.dim)
        self.phase_map = PhaseMap.load_from_hdf5(os.path.join(self.path, 'phase_map_ref.hdf5'))
        self.data.append(self.phase_map, self.projector)
        self.data.append(self.phase_map, self.projector)
        self.reg = FirstOrderRegularisator(self.mask, lam=1E-4)
        self.cost = Costfunction(ForwardModel(self.data), self.reg)

    def tearDown(self):
        self.path = None
        self.a = None
        self.dim = None
        self.mask = None
        self.data = None
        self.projector = None
        self.phase_map = None
        self.reg = None
        self.cost = None

    def test_call(self):
        assert_allclose(self.cost(np.ones(self.cost.n)), 0., atol=1E-7,
                        err_msg='Unexpected behaviour in __call__()!')
        zero_vec_cost = np.load(os.path.join(self.path, 'zero_vec_cost.npy'))
        assert_allclose(self.cost(np.zeros(self.cost.n)), zero_vec_cost,
                        err_msg='Unexpected behaviour in __call__()!')

    def test_jac(self):
        assert_allclose(self.cost.jac(np.ones(self.cost.n)), np.zeros(self.cost.n), atol=1E-7,
                        err_msg='Unexpected behaviour in jac()!')
        jac_vec_ref = np.load(os.path.join(self.path, 'jac_vec_ref.npy'))
        assert_allclose(self.cost.jac(np.zeros(self.cost.n)), jac_vec_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in jac()!')
        jac = np.array([self.cost.jac(np.eye(self.cost.n)[:, i]) for i in range(self.cost.n)]).T
        jac_ref = np.load(os.path.join(self.path, 'jac_ref.npy'))
        assert_allclose(jac, jac_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in jac()!')

    def test_hess_dot(self):
        assert_allclose(self.cost.hess_dot(None, np.zeros(self.cost.n)), np.zeros(self.cost.n),
                        atol=1E-7, err_msg='Unexpected behaviour in jac()!')
        hess_vec_ref = -np.load(os.path.join(self.path, 'jac_vec_ref.npy'))
        assert_allclose(self.cost.hess_dot(None, np.ones(self.cost.n)), hess_vec_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in jac()!')
        hess = np.array([self.cost.hess_dot(None, np.eye(self.cost.n)[:, i])
                         for i in range(self.cost.n)]).T
        hess_ref = np.load(os.path.join(self.path, 'hess_ref.npy'))
        assert_allclose(hess, hess_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in hess_dot()!')

    def test_hess_diag(self):
        assert_allclose(self.cost.hess_diag(None), np.ones(self.cost.n),
                        err_msg='Unexpected behaviour in hess_diag()!')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseCostfunction)
    unittest.TextTestRunner(verbosity=2).run(suite)

# -*- coding: utf-8 -*-
"""Testcase for the regularisator module"""


import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyramid.regularisator import NoneRegularisator
from pyramid.regularisator import ZeroOrderRegularisator
from pyramid.regularisator import FirstOrderRegularisator


class TestCaseNoneRegularisator(unittest.TestCase):

    def setUp(self):
        self.n = 9
        self.reg = NoneRegularisator()

    def tearDown(self):
        self.n = None
        self.reg = None

    def test_call(self):
        assert_allclose(self.reg(np.arange(self.n)), 0,
                        err_msg='Unexpected behaviour in __call__()!')

    def test_jac(self):
        assert_allclose(self.reg.jac(np.arange(self.n)), np.zeros(self.n),
                        err_msg='Unexpected behaviour in jac()!')

    def test_hess_dot(self):
        assert_allclose(self.reg.hess_dot(None, np.arange(self.n)), np.zeros(self.n),
                        err_msg='Unexpected behaviour in jac()!')

    def test_hess_diag(self):
        assert_allclose(self.reg.hess_diag(np.arange(self.n)), np.zeros(self.n),
                        err_msg='Unexpected behaviour in hess_diag()!')


class TestCaseZeroOrderRegularisator(unittest.TestCase):

    def setUp(self):
        self.n = 9
        self.lam = 1
        self.reg = ZeroOrderRegularisator(lam=self.lam)

    def tearDown(self):
        self.n = None
        self.lam = None
        self.reg = None

    def test_call(self):
        assert_allclose(self.reg(np.arange(self.n)), np.sum(np.arange(self.n)**2),
                        err_msg='Unexpected behaviour in __call__()!')

    def test_jac(self):
        assert_allclose(self.reg.jac(np.arange(self.n)), 2*np.arange(self.n),
                        err_msg='Unexpected behaviour in jac()!')
        jac = np.array([self.reg.jac(np.eye(self.n)[:, i]) for i in range(self.n)]).T
        assert_allclose(jac, 2*np.eye(self.n), err_msg='Unexpected behaviour in jac()!')

    def test_hess_dot(self):
        assert_allclose(self.reg.hess_dot(None, np.arange(self.n)), 2*np.arange(self.n),
                        err_msg='Unexpected behaviour in jac()!')
        hess = np.array([self.reg.hess_dot(None, np.eye(self.n)[:, i]) for i in range(self.n)]).T
        assert_allclose(hess, 2*np.eye(self.n), err_msg='Unexpected behaviour in hess_dot()!')

    def test_hess_diag(self):
        assert_allclose(self.reg.hess_diag(np.arange(self.n)), 2*np.ones(self.n),
                        err_msg='Unexpected behaviour in hess_diag()!')


class TestCaseFirstOrderRegularisator(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_regularisator')
        self.dim = (4, 5, 6)
        self.mask = np.zeros(self.dim, dtype=bool)
        self.mask[1:-1, 1:-1, 1:-1] = True
        self.n = 3 * self.mask.sum()
        self.lam = 1.
        self.reg = FirstOrderRegularisator(self.mask, lam=self.lam)

    def tearDown(self):
        self.path = None
        self.dim = None
        self.mask = None
        self.n = None
        self.lam = None
        self.reg = None

    def test_call(self):
        assert_allclose(self.reg(np.ones(self.n)), 0.,
                        err_msg='Unexpected behaviour in __call__()!')
        assert_allclose(self.reg(np.zeros(self.n)), 0.,
                        err_msg='Unexpected behaviour in __call__()!')
        cost_ref = np.load(os.path.join(self.path, 'first_order_cost_ref.npy'))
        assert_allclose(self.reg(np.arange(self.n)), cost_ref,
                        err_msg='Unexpected behaviour in __call__()!')

    def test_jac(self):
        assert_allclose(self.reg.jac(np.ones(self.n)), np.zeros(self.n),
                        err_msg='Unexpected behaviour in jac()!')
        assert_allclose(self.reg.jac(np.zeros(self.n)), np.zeros(self.n),
                        err_msg='Unexpected behaviour in jac()!')
        jac_vec_ref = np.load(os.path.join(self.path, 'first_order_jac_vec_ref.npy'))
        assert_allclose(self.reg.jac(np.arange(self.n)), jac_vec_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in jac()!')
        jac = np.array([self.reg.jac(np.eye(self.n)[:, i]) for i in range(self.n)]).T
        jac_ref = np.load(os.path.join(self.path, 'first_order_jac_ref.npy'))
        assert_allclose(jac, jac_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in jac()!')

    def test_hess_dot(self):
        assert_allclose(self.reg.hess_dot(None, np.ones(self.n)), np.zeros(self.n),
                        err_msg='Unexpected behaviour in hess_dot()!')
        assert_allclose(self.reg.hess_dot(None, np.zeros(self.n)), np.zeros(self.n),
                        err_msg='Unexpected behaviour in hess_dot()!')
        hess_vec_ref = np.load(os.path.join(self.path, 'first_order_jac_vec_ref.npy'))
        assert_allclose(self.reg.hess_dot(None, np.arange(self.n)), hess_vec_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in hess_dot()!')
        hess = np.array([self.reg.hess_dot(None, np.eye(self.n)[:, i]) for i in range(self.n)]).T
        hess_ref = np.load(os.path.join(self.path, 'first_order_jac_ref.npy'))
        assert_allclose(hess, hess_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in hess_dot()!')

    def test_hess_diag(self):
        hess_diag = self.reg.hess_diag(np.ones(self.n))
        hess_diag_ref = np.zeros(3*self.n)  # derivatives in all directions!
        first_order_jac_ref = np.load(os.path.join(self.path, 'first_order_jac_ref.npy'))
        hess_diag_ref[0:self.n] = np.diag(first_order_jac_ref)
        assert_allclose(hess_diag, hess_diag_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in hess_diag()!')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseNoneRegularisator)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseZeroOrderRegularisator)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseFirstOrderRegularisator)
    unittest.TextTestRunner(verbosity=2).run(suite)

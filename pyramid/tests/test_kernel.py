# -*- coding: utf-8 -*-
"""Testcase for the magdata module."""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyramid.kernel import Kernel


class TestCaseKernel(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_kernel')
        self.kernel = Kernel(1., dim_uv=(4, 4), b_0=1., geometry='disc')

    def tearDown(self):
        self.path = None
        self.kernel = None

    def test_kernel(self):
        ref_u = np.load(os.path.join(self.path, 'ref_u.npy'))
        ref_v = np.load(os.path.join(self.path, 'ref_v.npy'))
        ref_u_fft = np.load(os.path.join(self.path, 'ref_u_fft.npy'))
        ref_v_fft = np.load(os.path.join(self.path, 'ref_v_fft.npy'))
        assert_allclose(self.kernel.u, ref_u, err_msg='Unexpected behavior in u')
        assert_allclose(self.kernel.v, ref_v, err_msg='Unexpected behavior in v')
        assert_allclose(self.kernel.u_fft, ref_u_fft, atol=1E-7,
                        err_msg='Unexpected behavior in u_fft')
        assert_allclose(self.kernel.v_fft, ref_v_fft, atol=1E-7,
                        err_msg='Unexpected behavior in v_fft')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseKernel)
    unittest.TextTestRunner(verbosity=2).run(suite)

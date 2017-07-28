# -*- coding: utf-8 -*-
"""Testcase for the magdata module."""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyramid.kernel import Kernel, KernelCharge


class TestCaseKernel(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_kernel')
        self.kernel = Kernel(1., dim_uv=(4, 4), b_0=1., geometry='disc')

    def tearDown(self):
        self.path = None
        self.kernel = None

    def test_kernel(self):
        ref_u = np.load(os.path.join(self.path, 'ref_kc.npy'))
        ref_v = np.load(os.path.join(self.path, 'ref_v.npy'))
        ref_u_fft = np.load(os.path.join(self.path, 'ref_kc_fft.npy'))
        ref_v_fft = np.load(os.path.join(self.path, 'ref_v_fft.npy'))
        assert_allclose(self.kernel.u, ref_u, err_msg='Unexpected behavior in u')
        assert_allclose(self.kernel.v, ref_v, err_msg='Unexpected behavior in v')
        assert_allclose(self.kernel.u_fft, ref_u_fft, atol=1E-7,
                        err_msg='Unexpected behavior in u_fft')
        assert_allclose(self.kernel.v_fft, ref_v_fft, atol=1E-7,
                        err_msg='Unexpected behavior in v_fft')


class TestCaseKernelCharge(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_kernel')
        self.kernel = KernelCharge(1., dim_uv=(8, 8), electrode_vec=(3, 3))

    def tearDown(self):
        self.path = None
        self.kernel = None

    def test_kernelcharge(self):
        ref_kc = np.load(os.path.join(self.path, 'ref_kc.npy'))
        ref_kc_fft = np.load(os.path.join(self.path, 'ref_kc_fft.npy'))
        assert_allclose(self.kernel.kc, ref_kc, err_msg='Unexpected behavior in kc')
        assert_allclose(self.kernel.kc_fft, ref_kc_fft, atol=1E-7,
                        err_msg='Unexpected behavior in kc_fft')

# -*- coding: utf-8 -*-
"""Testcase for the magdata module."""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyramid.kernelcharge import KernelCharge


class TestCaseKernelCharge(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_kernelcharge')
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
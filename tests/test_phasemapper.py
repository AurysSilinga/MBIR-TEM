# -*- coding: utf-8 -*-
"""Testcase for the phasemapper module."""


import os
import unittest
import numpy as np

import pyramid.phasemapper as pm


class TestCasePhaseMapper(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_phasemapper')
        self.projection = (np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
        self.projection[0][1:-1, 1:-1] = 1
        self.projection[1][1:-1, 1:-1] = 1
        self.projection[2][1:-1, 1:-1] = 1
        self.res = 10.0

    def tearDown(self):
        self.path = None
        self.projection = None

    def test_phase_mag_fourier(self):
        phase = pm.phase_mag_fourier(self.res, self.projection, padding=10)
        reference = np.load(os.path.join(self.path, 'ref_phase_mag_fft.npy'))
        np.testing.assert_almost_equal(phase, reference, 7,
                                       'Unexpected behavior in phase_mag_fourier()')

    def test_phase_mag_real(self):
        phase_slab = pm.phase_mag_real(self.res, self.projection, method='slab')
        phase_disc = pm.phase_mag_real(self.res, self.projection, method='disc')
        ref_slab = np.load(os.path.join(self.path, 'ref_phase_mag_slab.npy'))
        ref_disc = np.load(os.path.join(self.path, 'ref_phase_mag_disc.npy'))
        np.testing.assert_almost_equal(phase_slab, ref_slab, 7,
                                       'Unexpected behavior in phase_mag_real() (slab)')
        np.testing.assert_almost_equal(phase_disc, ref_disc, 7,
                                       'Unexpected behavior in phase_mag_real() (disc)')

    def test_phase_mag_real_conv(self):
        phase_slab = pm.phase_mag_real_conv(self.res, self.projection, method='slab')
        phase_disc = pm.phase_mag_real_conv(self.res, self.projection, method='disc')
        ref_slab = np.load(os.path.join(self.path, 'ref_phase_mag_slab.npy'))
        ref_disc = np.load(os.path.join(self.path, 'ref_phase_mag_disc.npy'))
        np.testing.assert_almost_equal(phase_slab, ref_slab, 7,
                                       'Unexpected behavior in phase_mag_real_conv() (slab)')
        np.testing.assert_almost_equal(phase_disc, ref_disc, 7,
                                       'Unexpected behavior in phase_mag_real_conv() (disc)')

    def test_phase_elec(self):
        phase = pm.phase_elec(self.res, self.projection, v_0=1, v_acc=30000)
        reference = np.load(os.path.join(self.path, 'ref_phase_elec.npy'))
        np.testing.assert_almost_equal(phase, reference, 7,
                                       'Unexpected behavior in phase_elec()')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCasePhaseMapper)
    unittest.TextTestRunner(verbosity=2).run(suite)

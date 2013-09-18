# -*- coding: utf-8 -*-
"""Testcase for the analytic module."""


import os
import unittest
import numpy as np
from numpy import pi

import pyramid.analytic as an


class TestCaseAnalytic(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_analytic/')
        self.dim = (4, 4, 4)
        self.res = 10.0
        self.phi = pi/4
        self.center = (self.dim[0]/2-0.5, self.dim[1]/2-0.5, self.dim[2]/2-0.5)
        self.radius = self.dim[2]/4

    def tearDown(self):
        self.path = None
        self.dim = None
        self.res = None
        self.phi = None
        self.center = None
        self.radius = None

    def test_phase_mag_slab(self):
        width = (self.dim[0]/2, self.dim[1]/2, self.dim[2]/2)
        phase = an.phase_mag_slab(self.dim, self.res, self.phi, self.center, width)
        reference = np.load(os.path.join(self.path, 'ref_phase_slab.npy'))
        np.testing.assert_equal(phase, reference, 'Unexpected behavior in phase_mag_slab()')

    def test_phase_mag_disc(self):
        radius = self.dim[2]/4
        height = self.dim[2]/2
        phase = an.phase_mag_disc(self.dim, self.res, self.phi, self.center, radius, height)
        reference = np.load(os.path.join(self.path, 'ref_phase_disc.npy'))
        np.testing.assert_equal(phase, reference, 'Unexpected behavior in phase_mag_disc()')

    def test_phase_mag_sphere(self):
        radius = self.dim[2]/4
        phase = an.phase_mag_sphere(self.dim, self.res, self.phi, self.center, radius)
        reference = np.load(os.path.join(self.path, 'ref_phase_sphere.npy'))
        np.testing.assert_equal(phase, reference, 'Unexpected behavior in phase_mag_sphere()')

    def test_phase_mag_vortex(self):
        radius = self.dim[2]/4
        height = self.dim[2]/2
        phase = an.phase_mag_vortex(self.dim, self.res, self.center, radius, height)
        reference = np.load(os.path.join(self.path, 'ref_phase_vort.npy'))
        np.testing.assert_equal(phase, reference, 'Unexpected behavior in phase_mag_vortex()')

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseAnalytic)
    unittest.TextTestRunner(verbosity=2).run(suite)

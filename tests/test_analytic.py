# -*- coding: utf-8 -*-
"""Testcase for the analytic module."""

import os
import unittest

import numpy as np
from numpy import pi
from numpy.testing import assert_allclose

import pyramid.analytic as an


class TestCaseAnalytic(unittest.TestCase):
    """TestCase for the analytic module."""

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_analytic/')
    dim = (4, 4, 4)
    a = 10.0
    phi = pi / 4
    center = (dim[0] / 2, dim[1] / 2, dim[2] / 2)
    radius = dim[2] / 4

    def test_phase_mag_slab(self):
        """Test of the phase_mag_slab method."""
        width = (self.dim[0] / 2, self.dim[1] / 2, self.dim[2] / 2)
        phase = an.phase_mag_slab(self.dim, self.a, self.phi, self.center, width).phase
        reference = np.load(os.path.join(self.path, 'ref_phase_slab.npy'))
        assert_allclose(phase, reference, atol=1E-10,
                        err_msg='Unexpected behavior in phase_mag_slab()')

    def test_phase_mag_disc(self):
        """Test of the phase_mag_disc method."""
        radius = self.dim[2] / 4
        height = self.dim[2] / 2
        phase = an.phase_mag_disc(self.dim, self.a, self.phi, self.center, radius, height).phase
        reference = np.load(os.path.join(self.path, 'ref_phase_disc.npy'))
        assert_allclose(phase, reference, atol=1E-10,
                        err_msg='Unexpected behavior in phase_mag_disc()')

    def test_phase_mag_sphere(self):
        """Test of the phase_mag_sphere method."""
        radius = self.dim[2] / 4
        phase = an.phase_mag_sphere(self.dim, self.a, self.phi, self.center, radius).phase
        reference = np.load(os.path.join(self.path, 'ref_phase_sphere.npy'))
        assert_allclose(phase, reference, atol=1E-10,
                        err_msg='Unexpected behavior in phase_mag_sphere()')

    def test_phase_mag_vortex(self):
        """Test of the phase_mag_vortex method."""
        radius = self.dim[2] / 4
        height = self.dim[2] / 2
        phase = an.phase_mag_vortex(self.dim, self.a, self.center, radius, height).phase
        reference = np.load(os.path.join(self.path, 'ref_phase_vort.npy'))
        assert_allclose(phase, reference, atol=1E-10,
                        err_msg='Unexpected behavior in phase_mag_vortex()')

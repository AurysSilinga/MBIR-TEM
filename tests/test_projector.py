# -*- coding: utf-8 -*-
"""Testcase for the projector module."""


import os
import unittest
import numpy as np

import pyramid.projector as pj
from pyramid.magdata import MagData


class TestCaseProjector(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_projector')
        self.mag_data = MagData.load_from_llg(os.path.join(self.path, 'ref_mag_data.txt'))

    def tearDown(self):
        self.path = None
        self.mag_data = None

    def test_simple_axis_projection(self):
        z_proj_ref = (np.loadtxt(os.path.join(self.path, 'ref_proj_z.txt')))
        y_proj_ref = (np.loadtxt(os.path.join(self.path, 'ref_proj_y.txt')))
        x_proj_ref = (np.loadtxt(os.path.join(self.path, 'ref_proj_x.txt')))
        z_proj = pj.simple_axis_projection(self.mag_data, 'z')
        y_proj = pj.simple_axis_projection(self.mag_data, 'y')
        x_proj = pj.simple_axis_projection(self.mag_data, 'x')
        np.testing.assert_equal(z_proj[0], z_proj_ref, '1')
        np.testing.assert_equal(z_proj[1], z_proj_ref, '2')
        np.testing.assert_equal(z_proj[2], z_proj_ref, '3')
        np.testing.assert_equal(y_proj[0], y_proj_ref, '4')
        np.testing.assert_equal(y_proj[1], y_proj_ref, '5')
        np.testing.assert_equal(y_proj[2], y_proj_ref, '6')
        np.testing.assert_equal(x_proj[0], x_proj_ref, '7')
        np.testing.assert_equal(x_proj[1], x_proj_ref, '8')
        np.testing.assert_equal(x_proj[2], x_proj_ref, '9')

    def test_single_axis_projection(self):
        raise AssertionError


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseProjector)
    unittest.TextTestRunner(verbosity=2).run(suite)

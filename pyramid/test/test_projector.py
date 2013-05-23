# -*- coding: utf-8 -*-
"""Testcase for the projector module."""


import unittest
import numpy as np
from numpy import pi
import pyramid.projector as pj
import pyramid.magcreator as mc
from pyramid.magdata import MagData


class TestCaseProjector(unittest.TestCase):
    
    def setUp(self):
        self.mag_data = MagData.load_from_llg('test_projector/ref_mag_data.txt')
    
    def tearDown(self):
        self.mag_data = None
        
    def test_simple_axis_projection(self):
        z_proj_ref = (np.loadtxt('test_projector/ref_proj_z.txt'))
        y_proj_ref = (np.loadtxt('test_projector/ref_proj_y.txt'))
        x_proj_ref = (np.loadtxt('test_projector/ref_proj_x.txt'))
        z_proj = pj.simple_axis_projection(self.mag_data, 'z')
        y_proj = pj.simple_axis_projection(self.mag_data, 'y')
        x_proj = pj.simple_axis_projection(self.mag_data, 'x')
        np.testing.assert_equal(z_proj[0], z_proj_ref, 'Testmessage')
        np.testing.assert_equal(z_proj[1], z_proj_ref, 'Testmessage')
        np.testing.assert_equal(y_proj[0], y_proj_ref, 'Testmessage')
        np.testing.assert_equal(y_proj[1], y_proj_ref, 'Testmessage')
        np.testing.assert_equal(x_proj[0], x_proj_ref, 'Testmessage')
        np.testing.assert_equal(x_proj[1], x_proj_ref, 'Testmessage')
        
            
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseProjector)
    unittest.TextTestRunner(verbosity=2).run(suite)
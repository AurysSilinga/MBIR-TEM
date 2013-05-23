# -*- coding: utf-8 -*-
"""Testcase for the analytic module."""


import unittest
import pyramid.analytic as an


class TestCaseAnalytic(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
        
    def test_template(self):
        pass
    
    def test_phase_mag_slab(self):
        pass
    
    def test_phase_mag_disc(self):
        pass
    
    def test_phase_mag_sphere(self):
        pass
            
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseAnalytic)
    unittest.TextTestRunner(verbosity=2).run(suite)
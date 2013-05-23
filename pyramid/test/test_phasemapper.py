# -*- coding: utf-8 -*-
"""Testcase for the phasemapper module."""


import unittest
import pyramid.phasemapper as pm


class TestCasePhaseMapper(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
        
    def test_phase_mag_fourier(self):
        pass
    
    def test_phase_mag_real_slab(self):
        pass
    
    def test_phase_mag_real_disc(self):
        pass
        
            
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCasePhaseMapper)
    unittest.TextTestRunner(verbosity=2).run(suite)
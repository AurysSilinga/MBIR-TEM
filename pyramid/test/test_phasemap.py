# -*- coding: utf-8 -*-
"""Testcase for the phasemap module."""


import unittest
from pyramid.phasemap import PhaseMap


class TestCasePhaseMap(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
        
    def test_template(self):
        pass
    
    def test_load_from_txt(self):
        pass
    
    def test_save_to_txt(self):
        pass
    
    def test_load_from_netcdf(self):
        pass
    
    def test_save_to_netcdf(self):
        pass
    
    def test_display(self):
        pass
        
            
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCasePhaseMap)
    unittest.TextTestRunner(verbosity=2).run(suite)
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 07:10:28 2013

@author: Jan
"""
# py.test

import unittest


class TestSuite(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test(self):
        self.assertTrue(True)
        
    def test_almost(self):
        self.assertAlmostEqual(0, 0.01, places=1)
        
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSuite)
    unittest.TextTestRunner(verbosity=2).run(suite)
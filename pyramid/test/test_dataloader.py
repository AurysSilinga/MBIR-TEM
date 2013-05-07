# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 07:10:28 2013

@author: Jan
"""
# py.test

import unittest
import numpy as np
import pyramid.dataloader as dl
from numpy import pi


# TODO: define test constants somewhere
# TODO: proper error messages
# TODO: Docstring


class TestCaseDataloader(unittest.TestCase):
    
    def setUp(self):
        self.filename = 'test_dataloader/test_data.txt'
        self.mag_data = dl.MagDataLLG(self.filename)
    
    def tearDown(self):
        self.filename = None
        self.mag_data = None
        
    def test_filename(self):
        self.assertEqual(self.mag_data.filename, self.filename)
        
    def test_resolution(self):
        self.assertEqual(self.mag_data.res, 10.0)
        
    def test_dimensions(self):
        self.assertEqual(self.mag_data.dim, (1, 3, 5))
    
    def test_length(self):
        self.assertEqual(self.mag_data.length, (10.0, 30.0, 50.0))
        
    def test_magnitude(self):
        test_shape = (1, 3, 5)
        test_array = np.zeros(test_shape)
        z_mag = self.mag_data.magnitude[0]
        y_mag = self.mag_data.magnitude[1]
        x_mag = self.mag_data.magnitude[2]
        self.assertEqual(z_mag.shape, test_shape)
        self.assertEqual(y_mag.shape, test_shape)
        self.assertEqual(x_mag.shape, test_shape)
        np.testing.assert_array_equal(z_mag, test_array, 'Testmessage')
        test_array[:,1,1:4] = np.cos(pi/4)
        np.testing.assert_array_almost_equal(y_mag, test_array, err_msg='y failure')
        np.testing.assert_array_almost_equal(x_mag, test_array, err_msg='x failure')
        
        
        
        
        
    
#mc.create_hom_mag((3,5),10.0,pi/4,mc.slab,((1,2),(1,3)),'test.txt',plot_mag_distr=True)   
#        '{:7.6e}'
#        scale = 1.0E-9 / 1.0E-2  #from cm to nm
#        data = np.genfromtxt(filename, skip_header=2)
#        x_dim, y_dim, z_dim = np.genfromtxt(filename, dtype=int, 
#                                            skip_header=1, 
#                                            skip_footer=len(data[:, 0]))
#        res = (data[1, 0] - data[0, 0]) / scale
#        x_len, y_len, z_len = [data[-1, i]/scale+res/2 for i in range(3)]
#        x_mag, y_mag, z_mag = [data[:, i].reshape(z_dim, y_dim, x_dim).mean(0) 
#                               *z_len for i in range(3,6)]
#        #Reshape in Python and Igor is different, 
#        #Python fills rows first, Igor columns!
#        self.filename = filename
#        self.res = res
#        self.dim = (x_dim, y_dim, z_dim)
#        self.length = (x_len, y_len, z_len)
#        self.magnitude = (x_mag, y_mag, z_mag)
#        self.assertAlmostEqual(0, 0.01, places=1)
#        mc.create_hom_mag((10,10),10.0,0,mc.slab,((4,4),(5,5)),'test.txt')
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseDataloader)
    unittest.TextTestRunner(verbosity=2).run(suite)
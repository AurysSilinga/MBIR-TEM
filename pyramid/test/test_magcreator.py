# -*- coding: utf-8 -*-
"""Testcase for the magcreator module."""


import unittest
import numpy as np
import pyramid.magcreator as mc


# TODO: Proper messages


class TestCaseMagCreator(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
        
    def test_shape_slab(self):
        test_slab = mc.Shapes.slab((5,6,7), (2,3,4), (1,3,5))
        np.testing.assert_equal(test_slab, np.load('test_magcreator/ref_slab.npy'),
                         'Testmessage')
    
    def test_shape_disc(self):
        test_disc_z = mc.Shapes.disc((5,6,7), (2,3,4), 2, 3, 'z')
        test_disc_y = mc.Shapes.disc((5,6,7), (2,3,4), 2, 3, 'y')
        test_disc_x = mc.Shapes.disc((5,6,7), (2,3,4), 2, 3, 'x')
        np.testing.assert_equal(test_disc_z, np.load('test_magcreator/ref_disc_z.npy'),
                         'Testmessage')
        np.testing.assert_equal(test_disc_y, np.load('test_magcreator/ref_disc_y.npy'),
                         'Testmessage')
        np.testing.assert_equal(test_disc_x, np.load('test_magcreator/ref_disc_x.npy'),
                         'Testmessage')
                         
    def test_shape_sphere(self):
        test_sphere = mc.Shapes.sphere((5,6,7), (2,3,4), 2)
        np.testing.assert_equal(test_sphere, np.load('test_magcreator/ref_sphere.npy'),
                         'Testmessage')
    
    def test_shape_filament(self):
        test_filament_z = mc.Shapes.filament((5,6,7), (2,3), 'z')
        test_filament_y = mc.Shapes.filament((5,6,7), (2,3), 'y')
        test_filament_x = mc.Shapes.filament((5,6,7), (2,3), 'x')
        np.testing.assert_equal(test_filament_z, np.load('test_magcreator/ref_filament_z.npy'),
                         'Testmessage')
        np.testing.assert_equal(test_filament_y, np.load('test_magcreator/ref_filament_y.npy'),
                         'Testmessage')
        np.testing.assert_equal(test_filament_x, np.load('test_magcreator/ref_filament_x.npy'),
                         'Testmessage')
    
    def test_shape_pixel(self):
        test_pixel = mc.Shapes.pixel((5,6,7), (2,3,4))
        np.testing.assert_equal(test_pixel, np.load('test_magcreator/ref_pixel.npy'),
                         'Testmessage')
    
    def test_create_mag_dist(self):
        pass
    
    def test_create_mag_dist_comb(self):
        pass
        
            
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseMagCreator)
    unittest.TextTestRunner(verbosity=2).run(suite)
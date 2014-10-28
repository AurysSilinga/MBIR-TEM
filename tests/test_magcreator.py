# -*- coding: utf-8 -*-
"""Testcase for the magcreator module."""


import os
import unittest
import numpy as np
from numpy import pi

import pyramid.magcreator as mc


class TestCaseMagCreator(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_magcreator')

    def tearDown(self):
        self.path = None

    def test_shape_slab(self):
        test_slab = mc.Shapes.slab((5, 6, 7), (2, 3, 4), (1, 3, 5))
        np.testing.assert_almost_equal(test_slab, np.load(os.path.join(self.path, 'ref_slab.npy')),
                                err_msg='Created slab does not match expectation!')

    def test_shape_disc(self):
        test_disc_z = mc.Shapes.disc((5, 6, 7), (2, 3, 4), 2, 3, 'z')
        test_disc_y = mc.Shapes.disc((5, 6, 7), (2, 3, 4), 2, 3, 'y')
        test_disc_x = mc.Shapes.disc((5, 6, 7), (2, 3, 4), 2, 3, 'x')
        np.testing.assert_almost_equal(test_disc_z, np.load(os.path.join(self.path, 'ref_disc_z.npy')),
                                err_msg='Created disc in z-direction does not match expectation!')
        np.testing.assert_almost_equal(test_disc_y, np.load(os.path.join(self.path, 'ref_disc_y.npy')),
                                err_msg='Created disc in y-direction does not match expectation!')
        np.testing.assert_almost_equal(test_disc_x, np.load(os.path.join(self.path, 'ref_disc_x.npy')),
                                err_msg='Created disc in x-direction does not match expectation!')

    def test_shape_sphere(self):
        test_sphere = mc.Shapes.sphere((5, 6, 7), (2, 3, 4), 2)
        np.testing.assert_almost_equal(test_sphere, np.load(os.path.join(self.path, 'ref_sphere.npy')),
                                err_msg='Created sphere does not match expectation!')

    def test_shape_filament(self):
        test_filament_z = mc.Shapes.filament((5, 6, 7), (2, 3), 'z')
        test_filament_y = mc.Shapes.filament((5, 6, 7), (2, 3), 'y')
        test_filament_x = mc.Shapes.filament((5, 6, 7), (2, 3), 'x')
        np.testing.assert_almost_equal(test_filament_z, np.load(os.path.join(self.path, 'ref_fil_z.npy')),
                                err_msg='Created filament in z-direction does not match expectation!')
        np.testing.assert_almost_equal(test_filament_y, np.load(os.path.join(self.path, 'ref_fil_y.npy')),
                                err_msg='Created filament in y-direction does not match expectation!')
        np.testing.assert_almost_equal(test_filament_x, np.load(os.path.join(self.path, 'ref_fil_x.npy')),
                                err_msg='Created filament in x-direction does not match expectation!')

    def test_shape_pixel(self):
        test_pixel = mc.Shapes.pixel((5, 6, 7), (2, 3, 4))
        np.testing.assert_almost_equal(test_pixel, np.load(os.path.join(self.path, 'ref_pixel.npy')),
                                err_msg='Created pixel does not match expectation!')

    def test_create_mag_dist_homog(self):
        mag_shape = mc.Shapes.disc((1, 10, 10), (0, 4.5, 4.5), 3, 1)
        magnitude = mc.create_mag_dist_homog(mag_shape, pi/4)
        np.testing.assert_almost_equal(magnitude, np.load(os.path.join(self.path, 'ref_mag_disc.npy')),
                                err_msg='Created homog. magnetic distribution does not match expectation')

    def test_create_mag_dist_vortex(self):
        mag_shape = mc.Shapes.disc((1, 10, 10), (0, 4.5, 4.5), 3, 1)
        magnitude = mc.create_mag_dist_vortex(mag_shape)
        np.testing.assert_almost_equal(magnitude, np.load(os.path.join(self.path, 'ref_mag_vort.npy')),
                                err_msg='Created vortex magnetic distribution does not match expectation')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseMagCreator)
    unittest.TextTestRunner(verbosity=2).run(suite)

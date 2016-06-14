# -*- coding: utf-8 -*-
"""Testcase for the magcreator module."""

import os
import unittest

import numpy as np
from numpy import pi
from numpy.testing import assert_allclose

import pyramid.magcreator as mc
from pyramid import shapes


class TestCaseMagCreator(unittest.TestCase):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_magcreator')

    def test_shape_slab(self):
        test_slab = shapes.slab((5, 6, 7), (2.5, 3.5, 4.5), (1, 3, 5))
        assert_allclose(test_slab, np.load(os.path.join(self.path, 'ref_slab.npy')),
                        err_msg='Created slab does not match expectation!')

    def test_shape_disc(self):
        test_disc_z = shapes.disc((5, 6, 7), (2.5, 3.5, 4.5), 2, 3, 'z')
        test_disc_y = shapes.disc((5, 6, 7), (2.5, 3.5, 4.5), 2, 3, 'y')
        test_disc_x = shapes.disc((5, 6, 7), (2.5, 3.5, 4.5), 2, 3, 'x')
        assert_allclose(test_disc_z, np.load(os.path.join(self.path, 'ref_disc_z.npy')),
                        err_msg='Created disc in z-direction does not match expectation!')
        assert_allclose(test_disc_y, np.load(os.path.join(self.path, 'ref_disc_y.npy')),
                        err_msg='Created disc in y-direction does not match expectation!')
        assert_allclose(test_disc_x, np.load(os.path.join(self.path, 'ref_disc_x.npy')),
                        err_msg='Created disc in x-direction does not match expectation!')

    def test_shape_ellipse(self):
        test_ellipse_z = shapes.ellipse((7, 8, 9), (3.5, 4.5, 5.5), (3, 5), 1, 'z')
        test_ellipse_y = shapes.ellipse((7, 8, 9), (3.5, 4.5, 5.5), (3, 5), 1, 'y')
        test_ellipse_x = shapes.ellipse((7, 8, 9), (3.5, 4.5, 5.5), (3, 5), 1, 'x')
        assert_allclose(test_ellipse_z, np.load(os.path.join(self.path, 'ref_ellipse_z.npy')),
                        err_msg='Created ellipse does not match expectation (z)!')
        assert_allclose(test_ellipse_y, np.load(os.path.join(self.path, 'ref_ellipse_y.npy')),
                        err_msg='Created ellipse does not match expectation (y)!')
        assert_allclose(test_ellipse_x, np.load(os.path.join(self.path, 'ref_ellipse_x.npy')),
                        err_msg='Created ellipse does not match expectation (x)!')

    def test_shape_sphere(self):
        test_sphere = shapes.sphere((5, 6, 7), (2.5, 3.5, 4.5), 2)
        assert_allclose(test_sphere, np.load(os.path.join(self.path, 'ref_sphere.npy')),
                        err_msg='Created sphere does not match expectation!')

    def test_shape_ellipsoid(self):
        test_ellipsoid = shapes.ellipsoid((7, 8, 9), (3.5, 4.5, 4.5), (3, 5, 7))
        assert_allclose(test_ellipsoid, np.load(os.path.join(self.path, 'ref_ellipsoid.npy')),
                        err_msg='Created ellipsoid does not match expectation!')

    def test_shape_filament(self):
        test_filament_z = shapes.filament((5, 6, 7), (2, 3), 'z')
        test_filament_y = shapes.filament((5, 6, 7), (2, 3), 'y')
        test_filament_x = shapes.filament((5, 6, 7), (2, 3), 'x')
        assert_allclose(test_filament_z, np.load(os.path.join(self.path, 'ref_fil_z.npy')),
                        err_msg='Created filament in z-direction does not match expectation!')
        assert_allclose(test_filament_y, np.load(os.path.join(self.path, 'ref_fil_y.npy')),
                        err_msg='Created filament in y-direction does not match expectation!')
        assert_allclose(test_filament_x, np.load(os.path.join(self.path, 'ref_fil_x.npy')),
                        err_msg='Created filament in x-direction does not match expectation!')

    def test_shape_pixel(self):
        test_pixel = shapes.pixel((5, 6, 7), (2, 3, 4))
        assert_allclose(test_pixel, np.load(os.path.join(self.path, 'ref_pixel.npy')),
                        err_msg='Created pixel does not match expectation!')

    def test_create_mag_dist_homog(self):
        mag_shape = shapes.disc((1, 10, 10), (0, 5, 5), 3, 1)
        magnitude = mc.create_mag_dist_homog(mag_shape, pi / 4)
        assert_allclose(magnitude, np.load(os.path.join(self.path, 'ref_mag_disc.npy')),
                        err_msg='Created homog. magnetic distribution does not match expectation')

    def test_create_mag_dist_vortex(self):
        mag_shape = shapes.disc((1, 10, 10), (0, 5, 5), 3, 1)
        magnitude = mc.create_mag_dist_vortex(mag_shape)
        assert_allclose(magnitude, np.load(os.path.join(self.path, 'ref_mag_vort.npy')),
                        err_msg='Created vortex magnetic distribution does not match expectation')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseMagCreator)
    unittest.TextTestRunner(verbosity=2).run(suite)

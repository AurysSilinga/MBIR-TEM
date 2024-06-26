# -*- coding: utf-8 -*-
"""Testcase for the magcreator module."""

import os
import unittest

import numpy as np
from numpy import pi
from numpy.testing import assert_allclose

import pyramid.magcreator as mc


class TestCaseMagCreator(unittest.TestCase):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_magcreator')

    def test_shape_slab(self):
        test_slab = mc.shapes.slab((5, 6, 7), (2.5, 3.5, 4.5), (1, 3, 5))
        assert_allclose(test_slab, np.load(os.path.join(self.path, 'ref_slab.npy')),
                        err_msg='Created slab does not match expectation!')

    def test_shape_disc(self):
        test_disc_z = mc.shapes.disc((5, 6, 7), (2.5, 3.5, 4.5), 2, 3, 'z')
        test_disc_y = mc.shapes.disc((5, 6, 7), (2.5, 3.5, 4.5), 2, 3, 'y')
        test_disc_x = mc.shapes.disc((5, 6, 7), (2.5, 3.5, 4.5), 2, 3, 'x')
        assert_allclose(test_disc_z, np.load(os.path.join(self.path, 'ref_disc_z.npy')),
                        err_msg='Created disc in z-direction does not match expectation!')
        assert_allclose(test_disc_y, np.load(os.path.join(self.path, 'ref_disc_y.npy')),
                        err_msg='Created disc in y-direction does not match expectation!')
        assert_allclose(test_disc_x, np.load(os.path.join(self.path, 'ref_disc_x.npy')),
                        err_msg='Created disc in x-direction does not match expectation!')

    def test_shape_ellipse(self):
        test_ellipse_z = mc.shapes.ellipse((7, 8, 9), (3.5, 4.5, 5.5), (3, 5), 1, 'z')
        test_ellipse_y = mc.shapes.ellipse((7, 8, 9), (3.5, 4.5, 5.5), (3, 5), 1, 'y')
        test_ellipse_x = mc.shapes.ellipse((7, 8, 9), (3.5, 4.5, 5.5), (3, 5), 1, 'x')
        assert_allclose(test_ellipse_z, np.load(os.path.join(self.path, 'ref_ellipse_z.npy')),
                        err_msg='Created ellipse does not match expectation (z)!')
        assert_allclose(test_ellipse_y, np.load(os.path.join(self.path, 'ref_ellipse_y.npy')),
                        err_msg='Created ellipse does not match expectation (y)!')
        assert_allclose(test_ellipse_x, np.load(os.path.join(self.path, 'ref_ellipse_x.npy')),
                        err_msg='Created ellipse does not match expectation (x)!')

    def test_shape_sphere(self):
        test_sphere = mc.shapes.sphere((5, 6, 7), (2.5, 3.5, 4.5), 2)
        assert_allclose(test_sphere, np.load(os.path.join(self.path, 'ref_sphere.npy')),
                        err_msg='Created sphere does not match expectation!')

    def test_shape_ellipsoid(self):
        test_ellipsoid = mc.shapes.ellipsoid((7, 8, 9), (3.5, 4.5, 4.5), (3, 5, 7))
        assert_allclose(test_ellipsoid, np.load(os.path.join(self.path, 'ref_ellipsoid.npy')),
                        err_msg='Created ellipsoid does not match expectation!')

    def test_shape_filament(self):
        test_filament_z = mc.shapes.filament((5, 6, 7), (2, 3), 'z')
        test_filament_y = mc.shapes.filament((5, 6, 7), (2, 3), 'y')
        test_filament_x = mc.shapes.filament((5, 6, 7), (2, 3), 'x')
        assert_allclose(test_filament_z, np.load(os.path.join(self.path, 'ref_fil_z.npy')),
                        err_msg='Created filament in z-direction does not match expectation!')
        assert_allclose(test_filament_y, np.load(os.path.join(self.path, 'ref_fil_y.npy')),
                        err_msg='Created filament in y-direction does not match expectation!')
        assert_allclose(test_filament_x, np.load(os.path.join(self.path, 'ref_fil_x.npy')),
                        err_msg='Created filament in x-direction does not match expectation!')

    def test_shape_pixel(self):
        test_pixel = mc.shapes.pixel((5, 6, 7), (2, 3, 4))
        assert_allclose(test_pixel, np.load(os.path.join(self.path, 'ref_pixel.npy')),
                        err_msg='Created pixel does not match expectation!')

    def test_create_mag_dist_homog(self):
        mag_shape = mc.shapes.disc((1, 10, 10), (0, 5, 5), 3, 1)
        magnitude = mc.create_mag_dist_homog(mag_shape, pi / 4)
        assert_allclose(magnitude, np.load(os.path.join(self.path, 'ref_mag_disc.npy')),
                        err_msg='Created homog. magnetic distribution does not match expectation')

    def test_create_mag_dist_vortex(self):
        mag_shape = mc.shapes.disc((1, 10, 10), (0, 5, 5), 3, 1)
        magnitude = mc.create_mag_dist_vortex(mag_shape)
        assert_allclose(magnitude, np.load(os.path.join(self.path, 'ref_mag_vort.npy')),
                        err_msg='Created vortex magnetic distribution does not match expectation')

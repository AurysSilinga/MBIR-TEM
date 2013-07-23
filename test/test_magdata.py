# -*- coding: utf-8 -*-
"""Testcase for the magdata module."""


import unittest
import numpy as np
from numpy import pi
from pyramid.magdata import MagData

# py.test
# TODO: define test constants somewhere
# TODO: proper error messages
# TODO: Docstring


class TestCaseMagData(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        magnitude = (np.zeros((1, 1, 1), np.zeros(1, 1, 1), np.zeros(1, 1, 1)))
        magnitude = (np.zeros((1, 1, 1), np.zeros(1, 1, 1), np.zeros(1, 1, 1)))
        self.assertRaises(AssertionError, MagData, 10.0, )
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
        test_array[:, 1, 1:4] = np.cos(pi/4)
        np.testing.assert_array_almost_equal(y_mag, test_array, err_msg='y failure')
        np.testing.assert_array_almost_equal(x_mag, test_array, err_msg='x failure')

    def test_load_from_llg(self):
        pass

    def test_save_to_llg(self):
        pass

    def test_load_from_netcdf(self):
        pass

    def test_save_to_netcdf(self):
        pass

    def test_quiver_plot(self):
        pass

    def test_quiver_plot3d(self):
        pass

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseMagData)
    unittest.TextTestRunner(verbosity=2).run(suite)

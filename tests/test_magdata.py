# -*- coding: utf-8 -*-
"""Testcase for the magdata module."""


import os
import unittest
import numpy as np

from pyramid.magdata import MagData


class TestCaseMagData(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_magdata')
        magnitude = (np.zeros((4, 4, 4)), np.zeros((4, 4, 4)), np.zeros((4, 4, 4)))
        magnitude[0][1:-1, 1:-1, 1:-1] = 1
        magnitude[1][1:-1, 1:-1, 1:-1] = 1
        magnitude[2][1:-1, 1:-1, 1:-1] = 1
        self.mag_data = MagData(10.0, magnitude)

    def tearDown(self):
        self.path = None
        self.mag_data = None

    def test_add_magnitude(self):
        reference = (np.ones((4, 4, 4)), np.ones((4, 4, 4)), np.ones((4, 4, 4)))
        self.mag_data.add_magnitude(reference)
        reference[0][1:-1, 1:-1, 1:-1] = 2
        reference[1][1:-1, 1:-1, 1:-1] = 2
        reference[2][1:-1, 1:-1, 1:-1] = 2
        np.testing.assert_equal(self.mag_data.magnitude, reference,
                                'Unexpected behavior in add_magnitude()!')

    def test_get_mask(self):
        mask = self.mag_data.get_mask()
        reference = np.zeros((4, 4, 4))
        reference[1:-1, 1:-1, 1:-1] = True
        np.testing.assert_equal(mask, reference, 'Unexpected behavior in get_mask()!')

    def test_get_vector(self):
        mask = self.mag_data.get_mask()
        vector = self.mag_data.get_vector(mask)
        reference = np.ones(np.count_nonzero(self.mag_data.magnitude[0])*3)
        np.testing.assert_equal(vector, reference, 'Unexpected behavior in get_mask()!')

    def test_set_vector(self):
        mask = self.mag_data.get_mask()
        vector = 2 * np.ones(np.count_nonzero(self.mag_data.magnitude[0])*3)
        self.mag_data.set_vector(mask, vector)
        reference = (np.zeros((4, 4, 4)), np.zeros((4, 4, 4)), np.zeros((4, 4, 4)))
        reference[0][1:-1, 1:-1, 1:-1] = 2
        reference[1][1:-1, 1:-1, 1:-1] = 2
        reference[2][1:-1, 1:-1, 1:-1] = 2
        np.testing.assert_equal(self.mag_data.magnitude, reference,
                                'Unexpected behavior in set_mask()!')

    def test_scale_down(self):
        self.mag_data.scale_down()
        reference = (1/8. * np.ones((2, 2, 2)),
                     1/8. * np.ones((2, 2, 2)),
                     1/8. * np.ones((2, 2, 2)))
        np.testing.assert_equal(self.mag_data.magnitude, reference,
                                'Unexpected behavior in scale_down()!')
        np.testing.assert_equal(self.mag_data.res, 20, 'Unexpected behavior in scale_down()!')

    def test_load_from_llg(self):
        self.mag_data = MagData.load_from_llg(os.path.join(self.path, 'ref_mag_data.txt'))
        reference = (np.zeros((4, 4, 4)), np.zeros((4, 4, 4)), np.zeros((4, 4, 4)))
        reference[0][1:-1, 1:-1, 1:-1] = 1
        reference[1][1:-1, 1:-1, 1:-1] = 1
        reference[2][1:-1, 1:-1, 1:-1] = 1
        np.testing.assert_equal(self.mag_data.magnitude, reference,
                                'Unexpected behavior in load_from_llg()!')
        np.testing.assert_equal(self.mag_data.res, 10, 'Unexpected behavior in load_from_llg()!')

    def test_load_from_netcdf4(self):
        self.mag_data = MagData.load_from_netcdf4(os.path.join(self.path, 'ref_mag_data.nc'))
        reference = (np.zeros((4, 4, 4)), np.zeros((4, 4, 4)), np.zeros((4, 4, 4)))
        reference[0][1:-1, 1:-1, 1:-1] = 1
        reference[1][1:-1, 1:-1, 1:-1] = 1
        reference[2][1:-1, 1:-1, 1:-1] = 1
        np.testing.assert_equal(self.mag_data.magnitude, reference,
                                'Unexpected behavior in load_from_netcdf4()!')
        np.testing.assert_equal(self.mag_data.res, 10,
                                'Unexpected behavior in load_from_netcdf4()!')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseMagData)
    unittest.TextTestRunner(verbosity=2).run(suite)

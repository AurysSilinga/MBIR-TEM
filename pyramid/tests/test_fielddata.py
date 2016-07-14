# -*- coding: utf-8 -*-
"""Testcase for the magdata module."""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyramid.fielddata import VectorData


class TestCaseVectorData(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_fielddata')
        magnitude = np.zeros((3, 4, 4, 4))
        magnitude[:, 1:-1, 1:-1, 1:-1] = 1
        self.mag_data = VectorData(10.0, magnitude)

    def tearDown(self):
        self.path = None
        self.mag_data = None

    def test_copy(self):
        mag_data = self.mag_data
        mag_data_copy = self.mag_data.copy()
        assert mag_data == self.mag_data, 'Unexpected behaviour in copy()!'
        assert mag_data_copy != self.mag_data, 'Unexpected behaviour in copy()!'

    def test_scale_down(self):
        self.mag_data.scale_down()
        reference = 1 / 8. * np.ones((3, 2, 2, 2))
        assert_allclose(self.mag_data.field, reference,
                        err_msg='Unexpected behavior in scale_down()!')
        assert_allclose(self.mag_data.a, 20,
                        err_msg='Unexpected behavior in scale_down()!')

    def test_scale_up(self):
        self.mag_data.scale_up()
        reference = np.zeros((3, 8, 8, 8))
        reference[:, 2:6, 2:6, 2:6] = 1
        assert_allclose(self.mag_data.field, reference,
                        err_msg='Unexpected behavior in scale_down()!')
        assert_allclose(self.mag_data.a, 5,
                        err_msg='Unexpected behavior in scale_down()!')

    def test_pad(self):
        reference = self.mag_data.field.copy()
        self.mag_data.pad((1, 1, 1))
        reference = np.pad(reference, ((0, 0), (1, 1), (1, 1), (1, 1)), mode='constant')
        assert_allclose(self.mag_data.field, reference,
                        err_msg='Unexpected behavior in scale_down()!')
        self.mag_data.pad(((1, 1), (1, 1), (1, 1)))
        reference = np.pad(reference, ((0, 0), (1, 1), (1, 1), (1, 1)), mode='constant')
        assert_allclose(self.mag_data.field, reference,
                        err_msg='Unexpected behavior in scale_down()!')

    def test_get_mask(self):
        mask = self.mag_data.get_mask()
        reference = np.zeros((4, 4, 4))
        reference[1:-1, 1:-1, 1:-1] = True
        assert_allclose(mask, reference,
                        err_msg='Unexpected behavior in get_mask()!')

    def test_get_vector(self):
        mask = self.mag_data.get_mask()
        vector = self.mag_data.get_vector(mask)
        reference = np.ones(np.sum(mask) * 3)
        assert_allclose(vector, reference,
                        err_msg='Unexpected behavior in get_vector()!')

    def test_set_vector(self):
        mask = self.mag_data.get_mask()
        vector = 2 * np.ones(np.sum(mask) * 3)
        self.mag_data.set_vector(vector, mask)
        reference = np.zeros((3, 4, 4, 4))
        reference[:, 1:-1, 1:-1, 1:-1] = 2
        assert_allclose(self.mag_data.field, reference,
                        err_msg='Unexpected behavior in set_vector()!')

    def test_flip(self):
        mag_data = VectorData.load_from_hdf5(os.path.join(self.path, 'mag_data_orig.hdf5'))
        mag_data_flipx = VectorData.load_from_hdf5(os.path.join(self.path, 'mag_data_flipx.hdf5'))
        mag_data_flipy = VectorData.load_from_hdf5(os.path.join(self.path, 'mag_data_flipy.hdf5'))
        mag_data_flipz = VectorData.load_from_hdf5(os.path.join(self.path, 'mag_data_flipz.hdf5'))
        assert_allclose(mag_data.flip('x').field, mag_data_flipx.field,
                        err_msg='Unexpected behavior in flip()! (x)')
        assert_allclose(mag_data.flip('y').field, mag_data_flipy.field,
                        err_msg='Unexpected behavior in flip()! (y)')
        assert_allclose(mag_data.flip('z').field, mag_data_flipz.field,
                        err_msg='Unexpected behavior in flip()! (z)')

    def test_rot(self):
        mag_data = VectorData.load_from_hdf5(os.path.join(self.path, 'mag_data_orig.hdf5'))
        mag_data_rotx = VectorData.load_from_hdf5(os.path.join(self.path, 'mag_data_rotx.hdf5'))
        mag_data_roty = VectorData.load_from_hdf5(os.path.join(self.path, 'mag_data_roty.hdf5'))
        mag_data_rotz = VectorData.load_from_hdf5(os.path.join(self.path, 'mag_data_rotz.hdf5'))
        assert_allclose(mag_data.rot90('x').field, mag_data_rotx.field,
                        err_msg='Unexpected behavior in rot()! (x)')
        assert_allclose(mag_data.rot90('y').field, mag_data_roty.field,
                        err_msg='Unexpected behavior in rot()! (y)')
        assert_allclose(mag_data.rot90('z').field, mag_data_rotz.field,
                        err_msg='Unexpected behavior in rot()! (z)')

    def test_load_from_llg(self):
        mag_data = VectorData.load_from_llg(os.path.join(self.path, 'mag_data_ref_load.txt'))
        assert_allclose(mag_data.field, self.mag_data.field,
                        err_msg='Unexpected behavior in load_from_llg()!')
        assert_allclose(mag_data.a, self.mag_data.a,
                        err_msg='Unexpected behavior in load_from_llg()!')

    def test_load_from_hdf5(self):
        mag_data = VectorData.load_from_hdf5(os.path.join(self.path, 'mag_data_ref_load.hdf5'))
        assert_allclose(mag_data.field, self.mag_data.field,
                        err_msg='Unexpected behavior in load_from_hdf5()!')
        assert_allclose(mag_data.a, self.mag_data.a,
                        err_msg='Unexpected behavior in load_from_hdf5()!')


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)

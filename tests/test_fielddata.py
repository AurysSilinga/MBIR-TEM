# -*- coding: utf-8 -*-
"""Testcase for the magdata module."""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyramid.fielddata import VectorData
from pyramid import load_vectordata


class TestCaseVectorData(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_fielddata')
        magnitude = np.zeros((3, 4, 4, 4))
        magnitude[:, 1:-1, 1:-1, 1:-1] = 1
        self.magdata = VectorData(10.0, magnitude)

    def tearDown(self):
        self.path = None
        self.magdata = None

    def test_copy(self):
        magdata = self.magdata
        magdata_copy = self.magdata.copy()
        assert magdata == self.magdata, 'Unexpected behaviour in copy()!'
        assert magdata_copy != self.magdata, 'Unexpected behaviour in copy()!'

    def test_scale_down(self):
        magdata_test = self.magdata.scale_down()
        reference = 1 / 8. * np.ones((3, 2, 2, 2))
        assert_allclose(magdata_test.field, reference,
                        err_msg='Unexpected behavior in scale_down()!')
        assert_allclose(magdata_test.a, 20,
                        err_msg='Unexpected behavior in scale_down()!')

    def test_scale_up(self):
        magdata_test = self.magdata.scale_up()
        reference = np.zeros((3, 8, 8, 8))
        reference[:, 2:6, 2:6, 2:6] = 1
        assert_allclose(magdata_test.field, reference,
                        err_msg='Unexpected behavior in scale_up()!')
        assert_allclose(magdata_test.a, 5,
                        err_msg='Unexpected behavior in scale_up()!')

    def test_pad(self):
        magdata_test = self.magdata.pad((1, 1, 1))
        reference = self.magdata.field.copy()
        reference = np.pad(reference, ((0, 0), (1, 1), (1, 1), (1, 1)), mode='constant')
        assert_allclose(magdata_test.field, reference,
                        err_msg='Unexpected behavior in pad()!')
        magdata_test = magdata_test.pad(((1, 1), (1, 1), (1, 1)))
        reference = np.pad(reference, ((0, 0), (1, 1), (1, 1), (1, 1)), mode='constant')
        assert_allclose(magdata_test.field, reference,
                        err_msg='Unexpected behavior in pad()!')

    # TODO: Crop and several others are missing!

    def test_get_mask(self):
        mask = self.magdata.get_mask()
        reference = np.zeros((4, 4, 4))
        reference[1:-1, 1:-1, 1:-1] = True
        assert_allclose(mask, reference,
                        err_msg='Unexpected behavior in get_mask()!')

    def test_get_vector(self):
        mask = self.magdata.get_mask()
        vector = self.magdata.get_vector(mask)
        reference = np.ones(np.sum(mask) * 3)
        assert_allclose(vector, reference,
                        err_msg='Unexpected behavior in get_vector()!')

    def test_set_vector(self):
        mask = self.magdata.get_mask()
        vector = 2 * np.ones(np.sum(mask) * 3)
        self.magdata.set_vector(vector, mask)
        reference = np.zeros((3, 4, 4, 4))
        reference[:, 1:-1, 1:-1, 1:-1] = 2
        assert_allclose(self.magdata.field, reference,
                        err_msg='Unexpected behavior in set_vector()!')

    def test_flip(self):
        magdata = load_vectordata(os.path.join(self.path, 'magdata_orig.hdf5'))
        magdata_flipx = load_vectordata(os.path.join(self.path, 'magdata_flipx.hdf5'))
        magdata_flipy = load_vectordata(os.path.join(self.path, 'magdata_flipy.hdf5'))
        magdata_flipz = load_vectordata(os.path.join(self.path, 'magdata_flipz.hdf5'))
        assert_allclose(magdata.flip('x').field, magdata_flipx.field,
                        err_msg='Unexpected behavior in flip()! (x)')
        assert_allclose(magdata.flip('y').field, magdata_flipy.field,
                        err_msg='Unexpected behavior in flip()! (y)')
        assert_allclose(magdata.flip('z').field, magdata_flipz.field,
                        err_msg='Unexpected behavior in flip()! (z)')

    def test_rot(self):
        magdata = load_vectordata(os.path.join(self.path, 'magdata_orig.hdf5'))
        magdata_rotx = load_vectordata(os.path.join(self.path, 'magdata_rotx.hdf5'))
        magdata_roty = load_vectordata(os.path.join(self.path, 'magdata_roty.hdf5'))
        magdata_rotz = load_vectordata(os.path.join(self.path, 'magdata_rotz.hdf5'))
        assert_allclose(magdata.rot90('x').field, magdata_rotx.field,
                        err_msg='Unexpected behavior in rot()! (x)')
        assert_allclose(magdata.rot90('y').field, magdata_roty.field,
                        err_msg='Unexpected behavior in rot()! (y)')
        assert_allclose(magdata.rot90('z').field, magdata_rotz.field,
                        err_msg='Unexpected behavior in rot()! (z)')

    def test_load_from_llg(self):
        magdata = load_vectordata(os.path.join(self.path, 'magdata_ref_load.txt'))
        assert_allclose(magdata.field, self.magdata.field,
                        err_msg='Unexpected behavior in load_from_llg()!')
        assert_allclose(magdata.a, self.magdata.a,
                        err_msg='Unexpected behavior in load_from_llg()!')

    def test_load_from_hdf5(self):
        magdata = load_vectordata(os.path.join(self.path, 'magdata_ref_load.hdf5'))
        assert_allclose(magdata.field, self.magdata.field,
                        err_msg='Unexpected behavior in load_from_hdf5()!')
        assert_allclose(magdata.a, self.magdata.a,
                        err_msg='Unexpected behavior in load_from_hdf5()!')

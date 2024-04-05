# -*- coding: utf-8 -*-
"""Testcase for the phasemap module."""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyramid.phasemap import PhaseMap
from pyramid import load_phasemap


class TestCasePhaseMap(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_phasemap')
        phase = np.zeros((4, 4))
        phase[1:-1, 1:-1] = 1
        mask = phase.astype(dtype=np.bool)
        confidence = np.ones((4, 4))
        self.phasemap = PhaseMap(10.0, phase, mask, confidence)

    def tearDown(self):
        self.path = None
        self.phasemap = None

    def test_copy(self):
        phasemap = self.phasemap
        phasemap_copy = self.phasemap.copy()
        assert phasemap == self.phasemap, 'Unexpected behaviour in copy()!'
        assert phasemap_copy != self.phasemap, 'Unexpected behaviour in copy()!'

    def test_scale_down(self):
        phasemap_test = self.phasemap.scale_down()
        reference = 1 / 4. * np.ones((2, 2))
        assert_allclose(phasemap_test.phase, reference,
                        err_msg='Unexpected behavior in scale_down()!')
        assert_allclose(phasemap_test.mask, np.zeros((2, 2), dtype=np.bool),
                        err_msg='Unexpected behavior in scale_down()!')
        assert_allclose(phasemap_test.confidence, np.ones((2, 2)),
                        err_msg='Unexpected behavior in scale_down()!')
        assert_allclose(phasemap_test.a, 20,
                        err_msg='Unexpected behavior in scale_down()!')

    def test_scale_up(self):
        phasemap_test = self.phasemap.scale_up()
        reference = np.zeros((8, 8))
        reference[2:-2, 2:-2] = 1
        assert_allclose(phasemap_test.phase, reference,
                        err_msg='Unexpected behavior in scale_up()!')
        assert_allclose(phasemap_test.mask, reference.astype(dtype=np.bool),
                        err_msg='Unexpected behavior in scale_up()!')
        assert_allclose(phasemap_test.confidence, np.ones((8, 8)),
                        err_msg='Unexpected behavior in scale_up()!')
        assert_allclose(phasemap_test.a, 5,
                        err_msg='Unexpected behavior in scale_up()!')

    def test_load_from_txt(self):
        phasemap = load_phasemap(os.path.join(self.path, 'ref_phasemap.txt'))
        assert_allclose(self.phasemap.phase, phasemap.phase,
                        err_msg='Unexpected behavior in load_from_txt()!')
        assert_allclose(phasemap.a, self.phasemap.a,
                        err_msg='Unexpected behavior in load_from_txt()!')

    def test_load_from_hdf5(self):
        phasemap = load_phasemap(os.path.join(self.path, 'ref_phasemap.hdf5'))
        assert_allclose(self.phasemap.phase, phasemap.phase,
                        err_msg='Unexpected behavior in load_from_netcdf4()!')
        assert_allclose(self.phasemap.mask, phasemap.mask,
                        err_msg='Unexpected behavior in load_from_netcdf4()!')
        assert_allclose(self.phasemap.confidence, phasemap.confidence,
                        err_msg='Unexpected behavior in load_from_netcdf4()!')
        assert_allclose(phasemap.a, self.phasemap.a,
                        err_msg='Unexpected behavior in load_from_netcdf4()!')

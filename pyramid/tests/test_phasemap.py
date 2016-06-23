# -*- coding: utf-8 -*-
"""Testcase for the phasemap module."""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyramid.phasemap import PhaseMap


class TestCasePhaseMap(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_phasemap')
        phase = np.zeros((4, 4))
        phase[1:-1, 1:-1] = 1
        mask = phase.astype(dtype=np.bool)
        confidence = np.ones((4, 4))
        self.phase_map = PhaseMap(10.0, phase, mask, confidence)

    def tearDown(self):
        self.path = None
        self.phase_map = None

    def test_copy(self):
        phase_map = self.phase_map
        phase_map_copy = self.phase_map.copy()
        assert phase_map == self.phase_map, 'Unexpected behaviour in copy()!'
        assert phase_map_copy != self.phase_map, 'Unexpected behaviour in copy()!'

    def test_scale_down(self):
        self.phase_map.scale_down()
        reference = 1 / 4. * np.ones((2, 2))
        assert_allclose(self.phase_map.phase, reference,
                        err_msg='Unexpected behavior in scale_down()!')
        assert_allclose(self.phase_map.mask, np.zeros((2, 2), dtype=np.bool),
                        err_msg='Unexpected behavior in scale_down()!')
        assert_allclose(self.phase_map.confidence, np.ones((2, 2)),
                        err_msg='Unexpected behavior in scale_down()!')
        assert_allclose(self.phase_map.a, 20,
                        err_msg='Unexpected behavior in scale_down()!')

    def test_scale_up(self):
        self.phase_map.scale_up()
        reference = np.zeros((8, 8))
        reference[2:-2, 2:-2] = 1
        assert_allclose(self.phase_map.phase, reference,
                        err_msg='Unexpected behavior in scale_up()!')
        assert_allclose(self.phase_map.mask, reference.astype(dtype=np.bool),
                        err_msg='Unexpected behavior in scale_up()!')
        assert_allclose(self.phase_map.confidence, np.ones((8, 8)),
                        err_msg='Unexpected behavior in scale_up()!')
        assert_allclose(self.phase_map.a, 5,
                        err_msg='Unexpected behavior in scale_up()!')

    def test_load_from_txt(self):
        phase_map = PhaseMap.load_from_txt(os.path.join(self.path, 'ref_phase_map.txt'))
        assert_allclose(self.phase_map.phase, phase_map.phase,
                        err_msg='Unexpected behavior in load_from_txt()!')
        assert_allclose(phase_map.a, self.phase_map.a,
                        err_msg='Unexpected behavior in load_from_txt()!')

    def test_load_from_hdf5(self):
        phase_map = PhaseMap.load_from_hdf5(os.path.join(self.path, 'ref_phase_map.hdf5'))
        assert_allclose(self.phase_map.phase, phase_map.phase,
                        err_msg='Unexpected behavior in load_from_netcdf4()!')
        assert_allclose(self.phase_map.mask, phase_map.mask,
                        err_msg='Unexpected behavior in load_from_netcdf4()!')
        assert_allclose(self.phase_map.confidence, phase_map.confidence,
                        err_msg='Unexpected behavior in load_from_netcdf4()!')
        assert_allclose(phase_map.a, self.phase_map.a,
                        err_msg='Unexpected behavior in load_from_netcdf4()!')


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)

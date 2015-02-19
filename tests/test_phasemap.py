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
        self.phase_map = PhaseMap(10.0, phase)

    def tearDown(self):
        self.path = None
        self.phase_map = None

    def test_load_from_txt(self):
        phase_map = PhaseMap.load_from_txt(os.path.join(self.path, 'ref_phase_map.txt'))
        assert_allclose(phase_map.phase, self.phase_map.phase,
                        err_msg='Unexpected behavior in load_from_txt()!')
        assert_allclose(phase_map.a, self.phase_map.a,
                        err_msg='Unexpected behavior in load_from_txt()!')

    def test_load_from_netcdf4(self):
        phase_map = PhaseMap.load_from_netcdf4(os.path.join(self.path, 'ref_phase_map.nc'))
        assert_allclose(phase_map.phase, self.phase_map.phase,
                        err_msg='Unexpected behavior in load_from_netcdf4()!')
        assert_allclose(phase_map.a, self.phase_map.a,
                        err_msg='Unexpected behavior in load_from_netcdf4()!')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCasePhaseMap)
    unittest.TextTestRunner(verbosity=2).run(suite)

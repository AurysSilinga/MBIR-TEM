# -*- coding: utf-8 -*-
"""Testcase for the phasemap module."""


import os
import unittest
import numpy as np

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

    def test_set_unit(self):
        with self.assertRaises(AssertionError):
            self.phase_map.set_unit('nonsense')
        self.phase_map.set_unit('mrad')
        self.assertEqual(self.phase_map.unit, 'mrad', 'Unexpected behavior in set_unit()')

    def test_load_from_txt(self):
        self.mag_data = PhaseMap.load_from_txt(os.path.join(self.path, 'ref_phase_map.txt'))
        reference = np.zeros((4, 4))
        reference[1:-1, 1:-1] = 1
        np.testing.assert_equal(self.phase_map.phase, reference,
                                'Unexpected behavior in load_from_txt()!')
        np.testing.assert_equal(self.phase_map.res, 10, 'Unexpected behavior in load_from_llg()!')

    def test_load_from_netcdf4(self):
        self.mag_data = PhaseMap.load_from_netcdf4(os.path.join(self.path, 'ref_phase_map.nc'))
        reference = np.zeros((4, 4))
        reference[1:-1, 1:-1] = 1
        np.testing.assert_equal(self.phase_map.phase, reference,
                                'Unexpected behavior in load_from_netcdf4()!')
        np.testing.assert_equal(self.phase_map.res, 10,
                                'Unexpected behavior in load_from_netcdf4()!')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCasePhaseMap)
    unittest.TextTestRunner(verbosity=2).run(suite)

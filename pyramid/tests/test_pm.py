# -*- coding: utf-8 -*-
"""Testcase for the pm function."""

import os
import unittest

from numpy.testing import assert_allclose

from pyramid.fielddata import VectorData
from pyramid.phasemap import PhaseMap
from pyramid.utils import pm


class TestCasePM(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_phasemapper')
        self.mag_proj = VectorData.load_from_hdf5(os.path.join(self.path, 'mag_proj.hdf5'))

    def tearDown(self):
        self.path = None
        self.mag_proj = None
        self.mapper = None

    def test_pm(self):
        phase_ref = PhaseMap.load_from_hdf5(os.path.join(self.path, 'phase_map.hdf5'))
        phase_map = pm(self.mag_proj)
        assert_allclose(phase_map.phase, phase_ref.phase, atol=1E-7,
                        err_msg='Unexpected behavior in pm()!')
        assert_allclose(phase_map.a, phase_ref.a, err_msg='Unexpected behavior in pm()!')


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)

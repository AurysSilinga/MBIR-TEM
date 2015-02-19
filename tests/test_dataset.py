# -*- coding: utf-8 -*-
"""Testcase for the dataset module"""


import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyramid.dataset import DataSet
from pyramid.projector import SimpleProjector
from pyramid.phasemap import PhaseMap
from pyramid.magdata import MagData


class TestCaseDataSet(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_dataset')
        self.a = 10.
        self.dim = (4, 5, 6)
        self.mask = np.zeros(self.dim, dtype=bool)
        self.mask[1:-1, 1:-1, 1:-1] = True
        self.data = DataSet(self.a, self.dim, mask=self.mask)
        self.projector = SimpleProjector(self.dim)
        self.phase_map = PhaseMap(self.a, np.ones(self.dim[1:3]))

    def tearDown(self):
        self.path = None
        self.a = None
        self.dim = None
        self.mask = None
        self.data = None
        self.projector = None
        self.phase_map = None

    def test_append(self):
        self.data.append(self.phase_map, self.projector)
        assert self.data.phase_maps[0] == self.phase_map, 'Phase map not correctly assigned!'
        assert self.data.projectors[0] == self.projector, 'Projector not correctly assigned!'

    def test_create_phase_maps(self):
        self.data.projectors = [self.projector]
        mag_data = MagData(self.a, np.ones((3,)+self.dim))
        self.data.phase_maps = self.data.create_phase_maps(mag_data)
        phase_vec_ref = np.load(os.path.join(self.path, 'phase_vec_ref.npy'))
        assert_allclose(self.data.phase_vec, phase_vec_ref,
                        err_msg='Unexpected behaviour in create_phase_maps()!')

    def test_set_Se_inv_block_diag(self):
        self.data.append(self.phase_map, self.projector)
        self.data.append(self.phase_map, self.projector)
        cov = np.diag(np.ones(np.prod(self.phase_map.dim_uv)))
        self.data.set_Se_inv_block_diag([cov, cov])
        assert self.data.Se_inv.shape == (self.data.m, self.data.m), \
            'Unexpected behaviour in set_Se_inv_block_diag()!'
        assert self.data.Se_inv.diagonal().sum() == self.data.m, \
            'Unexpected behaviour in set_Se_inv_block_diag()!'

    def test_set_Se_inv_diag_with_masks(self):
        self.data.append(self.phase_map, self.projector)
        self.data.append(self.phase_map, self.projector)
        mask_2d = self.mask[1, ...]
        self.data.set_Se_inv_diag_with_masks([mask_2d, mask_2d])
        assert self.data.Se_inv.shape == (self.data.m, self.data.m), \
            'Unexpected behaviour in set_Se_inv_diag_with_masks()!'
        assert self.data.Se_inv.diagonal().sum() == 2*mask_2d.sum(), \
            'Unexpected behaviour in set_Se_inv_diag_with_masks()!'

    def test_create_3d_mask(self):
        self.assertRaises(NotImplementedError, self.data.create_3d_mask, None)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseDataSet)
    unittest.TextTestRunner(verbosity=2).run(suite)

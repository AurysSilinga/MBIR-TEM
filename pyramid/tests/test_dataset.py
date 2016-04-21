# -*- coding: utf-8 -*-
"""Testcase for the dataset module"""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyramid.dataset import DataSet
from pyramid.fielddata import VectorData
from pyramid.phasemap import PhaseMap
from pyramid.projector import SimpleProjector


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
        mag_data = VectorData(self.a, np.ones((3,) + self.dim))
        self.data.phase_maps = self.data.create_phase_maps(mag_data)
        phase_vec_ref = np.load(os.path.join(self.path, 'phase_vec_ref.npy'))
        assert_allclose(self.data.phase_vec, phase_vec_ref, atol=1E-6,
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

    def test_set_Se_inv_diag_with_conf(self):
        self.data.append(self.phase_map, self.projector)
        self.data.append(self.phase_map, self.projector)
        confidence = self.mask[1, ...]
        self.data.set_Se_inv_diag_with_conf([confidence, confidence])
        assert self.data.Se_inv.shape == (self.data.m, self.data.m), \
            'Unexpected behaviour in set_Se_inv_diag_with_masks()!'
        assert self.data.Se_inv.diagonal().sum() == 2 * confidence.sum(), \
            'Unexpected behaviour in set_Se_inv_diag_with_masks()!'

    def test_set_3d_mask(self):
        projector_z = SimpleProjector(self.dim, axis='z')
        projector_y = SimpleProjector(self.dim, axis='y')
        projector_x = SimpleProjector(self.dim, axis='x')
        mask_z = np.zeros(projector_z.dim_uv, dtype=bool)
        mask_y = np.zeros(projector_y.dim_uv, dtype=bool)
        mask_x = np.zeros(projector_x.dim_uv, dtype=bool)
        mask_z[1:-1, 1:-1] = True
        mask_y[1:-1, 1:-1] = True
        mask_x[1:-1, 1:-1] = True
        phase_map_z = PhaseMap(self.a, np.zeros(projector_z.dim_uv), mask_z)
        phase_map_y = PhaseMap(self.a, np.zeros(projector_y.dim_uv), mask_y)
        phase_map_x = PhaseMap(self.a, np.zeros(projector_x.dim_uv), mask_x)
        self.data.append(phase_map_z, projector_z)
        self.data.append(phase_map_y, projector_y)
        self.data.append(phase_map_x, projector_x)
        self.data.set_3d_mask()
        mask_ref = np.zeros(self.dim, dtype=bool)
        mask_ref[1:-1, 1:-1, 1:-1] = True
        np.testing.assert_equal(self.data.mask, mask_ref,
                                err_msg='Unexpected behaviour in set_3d_mask')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseDataSet)
    unittest.TextTestRunner(verbosity=2).run(suite)

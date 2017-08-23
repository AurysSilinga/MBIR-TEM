# -*- coding: utf-8 -*-
"""Testcase for the dataset module"""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyramid.dataset import DataSet, DataSetCharge
from pyramid.fielddata import VectorData, ScalarData
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
        self.phasemap = PhaseMap(self.a, np.ones(self.dim[1:3]))

    def tearDown(self):
        self.path = None
        self.a = None
        self.dim = None
        self.mask = None
        self.data = None
        self.projector = None
        self.phasemap = None

    def test_append(self):
        self.data.append(self.phasemap, self.projector)
        assert self.data.phasemaps[0] == self.phasemap, 'Phase map not correctly assigned!'
        assert self.data.projectors[0] == self.projector, 'Projector not correctly assigned!'

    def test_create_phasemaps(self):
        self.data.append(PhaseMap(self.a, np.zeros(self.projector.dim_uv)), self.projector)
        magdata = VectorData(self.a, np.ones((3,) + self.dim))
        phasemaps = self.data.create_phasemaps(magdata)
        phase_vec = phasemaps[0].phase_vec
        phase_vec_ref = np.load(os.path.join(self.path, 'phase_vec_ref.npy'))
        assert_allclose(phase_vec, phase_vec_ref, atol=1E-6,
                        err_msg='Unexpected behaviour in create_phasemaps()!')

    def test_set_Se_inv_block_diag(self):
        self.data.append(self.phasemap, self.projector)
        self.data.append(self.phasemap, self.projector)
        cov = np.diag(np.ones(np.prod(self.phasemap.dim_uv)))
        self.data.set_Se_inv_block_diag([cov, cov])
        assert self.data.Se_inv.shape == (self.data.m, self.data.m), \
            'Unexpected behaviour in set_Se_inv_block_diag()!'
        assert self.data.Se_inv.diagonal().sum() == self.data.m, \
            'Unexpected behaviour in set_Se_inv_block_diag()!'

    def test_set_Se_inv_diag_with_conf(self):
        self.data.append(self.phasemap, self.projector)
        self.data.append(self.phasemap, self.projector)
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
        phasemap_z = PhaseMap(self.a, np.zeros(projector_z.dim_uv), mask_z)
        phasemap_y = PhaseMap(self.a, np.zeros(projector_y.dim_uv), mask_y)
        phasemap_x = PhaseMap(self.a, np.zeros(projector_x.dim_uv), mask_x)
        self.data.append(phasemap_z, projector_z)
        self.data.append(phasemap_y, projector_y)
        self.data.append(phasemap_x, projector_x)
        self.data.set_3d_mask()
        mask_ref = np.zeros(self.dim, dtype=bool)
        mask_ref[1:-1, 1:-1, 1:-1] = True
        np.testing.assert_equal(self.data.mask, mask_ref,
                                err_msg='Unexpected behaviour in set_3d_mask')


class TestCaseDataSetCharge(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_dataset')
        self.a = 10.
        self.dim = (4, 5, 6)
        self.electrode_vec = (1E6, 1E6)
        self.mask = np.zeros(self.dim, dtype=bool)
        self.mask[1:-1, 1:-1, 1:-1] = True
        self.data = DataSetCharge(self.a, self.dim, self.electrode_vec, mask=self.mask)
        self.projector = SimpleProjector(self.dim)
        self.phasemap = PhaseMap(self.a, np.ones(self.dim[1:3]))

    def tearDown(self):
        self.path = None
        self.a = None
        self.dim = None
        self.mask = None
        self.data = None
        self.projector = None
        self.phasemap = None

    def test_append(self):
        self.data.append(self.phasemap, self.projector)
        assert self.data.phasemaps[0] == self.phasemap, 'Phase map not correctly assigned!'
        assert self.data.projectors[0] == self.projector, 'Projector not correctly assigned!'

    def test_create_phasemaps(self):
        self.data.append(PhaseMap(self.a, np.zeros(self.projector.dim_uv)), self.projector)
        elecdata = ScalarData(self.a, np.ones(self.dim))
        phasemaps = self.data.create_phasemaps(elecdata)
        phase_vec = phasemaps[0].phase_vec
        phase_vec_ref = np.load(os.path.join(self.path, 'charge_phase_vec_ref.npy'))
        assert_allclose(phase_vec, phase_vec_ref, atol=1E-6,
                        err_msg='Unexpected behaviour in create_phasemaps()!')

    def test_set_Se_inv_block_diag(self):
        self.data.append(self.phasemap, self.projector)
        self.data.append(self.phasemap, self.projector)
        cov = np.diag(np.ones(np.prod(self.phasemap.dim_uv)))
        self.data.set_Se_inv_block_diag([cov, cov])
        assert self.data.Se_inv.shape == (self.data.m, self.data.m), \
            'Unexpected behaviour in set_Se_inv_block_diag()!'
        assert self.data.Se_inv.diagonal().sum() == self.data.m, \
            'Unexpected behaviour in set_Se_inv_block_diag()!'

    def test_set_Se_inv_diag_with_conf(self):
        self.data.append(self.phasemap, self.projector)
        self.data.append(self.phasemap, self.projector)
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
        phasemap_z = PhaseMap(self.a, np.zeros(projector_z.dim_uv), mask_z)
        phasemap_y = PhaseMap(self.a, np.zeros(projector_y.dim_uv), mask_y)
        phasemap_x = PhaseMap(self.a, np.zeros(projector_x.dim_uv), mask_x)
        self.data.append(phasemap_z, projector_z)
        self.data.append(phasemap_y, projector_y)
        self.data.append(phasemap_x, projector_x)
        self.data.set_3d_mask()
        mask_ref = np.zeros(self.dim, dtype=bool)
        mask_ref[1:-1, 1:-1, 1:-1] = True
        np.testing.assert_equal(self.data.mask, mask_ref,
                                err_msg='Unexpected behaviour in set_3d_mask')
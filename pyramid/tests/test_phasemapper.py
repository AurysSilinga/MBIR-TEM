# -*- coding: utf-8 -*-
"""Testcase for the phasemapper module."""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyramid.kernel import Kernel
from pyramid.fielddata import VectorData, ScalarData
from pyramid.phasemap import PhaseMap
from pyramid.phasemapper import PhaseMapperRDFC, PhaseMapperRDRC, PhaseMapperFDFC
from pyramid.phasemapper import PhaseMapperMIP, pm


class TestCasePhaseMapperRDFC(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_phasemapper')
        self.mag_proj = VectorData.load_from_hdf5(os.path.join(self.path, 'mag_proj.hdf5'))
        self.mapper = PhaseMapperRDFC(Kernel(self.mag_proj.a, self.mag_proj.dim[1:]))

    def tearDown(self):
        self.path = None
        self.mag_proj = None
        self.mapper = None

    def test_PhaseMapperRDFC_call(self):
        phase_ref = PhaseMap.load_from_hdf5(os.path.join(self.path, 'phase_map.hdf5'))
        phase_map = self.mapper(self.mag_proj)
        assert_allclose(phase_map.phase, phase_ref.phase, atol=1E-7,
                        err_msg='Unexpected behavior in __call__()!')
        assert_allclose(phase_map.a, phase_ref.a, err_msg='Unexpected behavior in __call__()!')

    def test_PhaseMapperRDFC_jac_dot(self):
        phase = self.mapper(self.mag_proj).phase
        mag_proj_vec = self.mag_proj.field[:2, ...].flatten()
        phase_jac = self.mapper.jac_dot(mag_proj_vec).reshape(self.mapper.kernel.dim_uv)
        assert_allclose(phase, phase_jac, atol=1E-7,
                        err_msg='Inconsistency between __call__() and jac_dot()!')
        n = self.mapper.n
        jac = np.array([self.mapper.jac_dot(np.eye(n)[:, i]) for i in range(n)]).T
        jac_ref = np.load(os.path.join(self.path, 'jac.npy'))
        assert_allclose(jac, jac_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in the the jacobi matrix!')

    def test_PhaseMapperRDFC_jac_T_dot(self):
        m = self.mapper.m
        jac_T = np.array([self.mapper.jac_T_dot(np.eye(m)[:, i]) for i in range(m)]).T
        jac_T_ref = np.load(os.path.join(self.path, 'jac.npy')).T
        assert_allclose(jac_T, jac_T_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in the the transposed jacobi matrix!')


class TestCasePhaseMapperRDRC(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_phasemapper')
        self.mag_proj = VectorData.load_from_hdf5(os.path.join(self.path, 'mag_proj.hdf5'))
        self.mapper = PhaseMapperRDRC(Kernel(self.mag_proj.a, self.mag_proj.dim[1:]))

    def tearDown(self):
        self.path = None
        self.mag_proj = None
        self.mapper = None

    def test_PhaseMapperRDRC_call(self):
        phase_ref = PhaseMap.load_from_hdf5(os.path.join(self.path, 'phase_map.hdf5'))
        phase_map = self.mapper(self.mag_proj)
        assert_allclose(phase_map.phase, phase_ref.phase, atol=1E-7,
                        err_msg='Unexpected behavior in __call__()!')
        assert_allclose(phase_map.a, phase_ref.a, err_msg='Unexpected behavior in __call__()!')

    def test_PhaseMapperRDRC_jac_dot(self):
        phase = self.mapper(self.mag_proj).phase
        mag_proj_vec = self.mag_proj.field[:2, ...].flatten()
        phase_jac = self.mapper.jac_dot(mag_proj_vec).reshape(self.mapper.kernel.dim_uv)
        assert_allclose(phase, phase_jac, atol=1E-7,
                        err_msg='Inconsistency between __call__() and jac_dot()!')
        n = self.mapper.n
        jac = np.array([self.mapper.jac_dot(np.eye(n)[:, i]) for i in range(n)]).T
        jac_ref = np.load(os.path.join(self.path, 'jac.npy'))
        assert_allclose(jac, jac_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in the the jacobi matrix!')

    def test_PhaseMapperRDRC_jac_T_dot(self):
        m = self.mapper.m
        jac_T = np.array([self.mapper.jac_T_dot(np.eye(m)[:, i]) for i in range(m)]).T
        jac_T_ref = np.load(os.path.join(self.path, 'jac.npy')).T
        assert_allclose(jac_T, jac_T_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in the the transposed jacobi matrix!')


class TestCasePhaseMapperFDFCpad0(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_phasemapper')
        self.mag_proj = VectorData.load_from_hdf5(os.path.join(self.path, 'mag_proj.hdf5'))
        self.mapper = PhaseMapperFDFC(self.mag_proj.a, self.mag_proj.dim[1:], padding=0)

    def tearDown(self):
        self.path = None
        self.mag_proj = None
        self.mapper = None

    def test_PhaseMapperFDFC_call(self):
        phase_ref = PhaseMap.load_from_hdf5(os.path.join(self.path, 'phase_map_fc.hdf5'))
        phase_map = self.mapper(self.mag_proj)
        assert_allclose(phase_map.phase, phase_ref.phase, atol=1E-7,
                        err_msg='Unexpected behavior in __call__()!')
        assert_allclose(phase_map.a, phase_ref.a, err_msg='Unexpected behavior in __call__()!')

    def test_PhaseMapperFDFC_jac_dot(self):
        phase = self.mapper(self.mag_proj).phase
        mag_proj_vec = self.mag_proj.field[:2, ...].flatten()
        phase_jac = self.mapper.jac_dot(mag_proj_vec).reshape(self.mapper.dim_uv)
        assert_allclose(phase, phase_jac, atol=1E-7,
                        err_msg='Inconsistency between __call__() and jac_dot()!')
        n = self.mapper.n
        jac = np.array([self.mapper.jac_dot(np.eye(n)[:, i]) for i in range(n)]).T
        jac_ref = np.load(os.path.join(self.path, 'jac_fc.npy'))
        assert_allclose(jac, jac_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in the the jacobi matrix!')

    def test_PhaseMapperFDFC_jac_T_dot(self):
        self.assertRaises(NotImplementedError, self.mapper.jac_T_dot, None)


class TestCasePhaseMapperFDFCpad1(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_phasemapper')
        self.mag_proj = VectorData.load_from_hdf5(os.path.join(self.path, 'mag_proj.hdf5'))
        self.mapper = PhaseMapperFDFC(self.mag_proj.a, self.mag_proj.dim[1:], padding=1)

    def tearDown(self):
        self.path = None
        self.mag_proj = None
        self.mapper = None

    def test_PhaseMapperFDFC_call(self):
        phase_ref = PhaseMap.load_from_hdf5(os.path.join(self.path, 'phase_map_fc_pad.hdf5'))
        phase_map = self.mapper(self.mag_proj)
        assert_allclose(phase_map.phase, phase_ref.phase, atol=1E-7,
                        err_msg='Unexpected behavior in __call__()!')
        assert_allclose(phase_map.a, phase_ref.a, err_msg='Unexpected behavior in __call__()!')

    def test_PhaseMapperFDFC_jac_dot(self):
        phase = self.mapper(self.mag_proj).phase
        mag_proj_vec = self.mag_proj.field[:2, ...].flatten()
        phase_jac = self.mapper.jac_dot(mag_proj_vec).reshape(self.mapper.dim_uv)
        assert_allclose(phase, phase_jac, atol=1E-7,
                        err_msg='Inconsistency between __call__() and jac_dot()!')
        n = self.mapper.n
        jac = np.array([self.mapper.jac_dot(np.eye(n)[:, i]) for i in range(n)]).T
        jac_ref = np.load(os.path.join(self.path, 'jac_fc_pad.npy'))
        assert_allclose(jac, jac_ref, atol=1E-7,
                        err_msg='Unexpected behaviour in the the jacobi matrix!')

    def test_PhaseMapperFDFC_jac_T_dot(self):
        self.assertRaises(NotImplementedError, self.mapper.jac_T_dot, None)


class TestCasePhaseMapperMIP(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_phasemapper')
        self.elec_proj = ScalarData.load_from_hdf5(os.path.join(self.path, 'elec_proj.hdf5'))
        self.mapper = PhaseMapperMIP(self.elec_proj.a, self.elec_proj.dim[1:])

    def tearDown(self):
        self.path = None
        self.elec_proj = None
        self.mapper = None

    def test_call(self):
        phase_ref = PhaseMap.load_from_hdf5(os.path.join(self.path, 'phase_map_elec.hdf5'))
        phase_map = self.mapper(self.elec_proj)
        assert_allclose(phase_map.phase, phase_ref.phase, atol=1E-7,
                        err_msg='Unexpected behavior in __call__()!')
        assert_allclose(phase_map.a, phase_ref.a, err_msg='Unexpected behavior in __call__()!')

    def test_jac_dot(self):
        self.assertRaises(NotImplementedError, self.mapper.jac_dot, None)

    def test_jac_T_dot(self):
        self.assertRaises(NotImplementedError, self.mapper.jac_T_dot, None)


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

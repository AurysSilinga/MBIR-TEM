# -*- coding: utf-8 -*-
"""Testcase for the projector module."""

import os
import unittest

import numpy as np
from numpy import pi
from numpy.testing import assert_allclose

from pyramid.projector import XTiltProjector, YTiltProjector, SimpleProjector
from pyramid import load_vectordata


class TestCaseSimpleProjector(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_projector')
        self.magdata = load_vectordata(os.path.join(self.path, 'ref_magdata.hdf5'))
        self.proj_z = SimpleProjector(self.magdata.dim, axis='z')
        self.proj_y = SimpleProjector(self.magdata.dim, axis='y')
        self.proj_x = SimpleProjector(self.magdata.dim, axis='x')

    def tearDown(self):
        self.path = None
        self.magdata = None
        self.proj_z = None
        self.proj_y = None
        self.proj_x = None

    def test_SimpleProjector_call(self):
        mag_proj_z = self.proj_z(self.magdata)
        mag_proj_y = self.proj_y(self.magdata)
        mag_proj_x = self.proj_x(self.magdata)
        mag_proj_z_ref = load_vectordata(os.path.join(self.path, 'ref_mag_proj_z.hdf5'))
        mag_proj_y_ref = load_vectordata(os.path.join(self.path, 'ref_mag_proj_y.hdf5'))
        mag_proj_x_ref = load_vectordata(os.path.join(self.path, 'ref_mag_proj_x.hdf5'))
        assert_allclose(mag_proj_z.field, mag_proj_z_ref.field,
                        err_msg='Unexpected behaviour in SimpleProjector (z-axis)')
        assert_allclose(mag_proj_y.field, mag_proj_y_ref.field,
                        err_msg='Unexpected behaviour in SimpleProjector (y-axis)')
        assert_allclose(mag_proj_x.field, mag_proj_x_ref.field,
                        err_msg='Unexpected behaviour in SimpleProjector (x-axis)')

    def test_SimpleProjector_jac_dot(self):
        mag_vec = self.magdata.field_vec
        mag_proj_z = self.proj_z.jac_dot(mag_vec).reshape((2,) + self.proj_z.dim_uv)
        mag_proj_y = self.proj_y.jac_dot(mag_vec).reshape((2,) + self.proj_y.dim_uv)
        mag_proj_x = self.proj_x.jac_dot(mag_vec).reshape((2,) + self.proj_x.dim_uv)
        mag_proj_z_ref = load_vectordata(os.path.join(self.path, 'ref_mag_proj_z.hdf5'))
        mag_proj_y_ref = load_vectordata(os.path.join(self.path, 'ref_mag_proj_y.hdf5'))
        mag_proj_x_ref = load_vectordata(os.path.join(self.path, 'ref_mag_proj_x.hdf5'))
        assert_allclose(mag_proj_z, mag_proj_z_ref.field[:2, 0, ...],
                        err_msg='Inconsistency between __call__() and jac_dot()! (z-axis)')
        assert_allclose(mag_proj_y, mag_proj_y_ref.field[:2, 0, ...],
                        err_msg='Inconsistency between __call__() and jac_dot()! (y-axis)')
        assert_allclose(mag_proj_x, mag_proj_x_ref.field[:2, 0, ...],
                        err_msg='Inconsistency between __call__() and jac_dot()! (x-axis)')
        nz = self.proj_z.n
        ny = self.proj_y.n
        nx = self.proj_x.n
        jac_z = np.array([self.proj_z.jac_dot(np.eye(nz)[:, i]) for i in range(nz)]).T
        jac_y = np.array([self.proj_y.jac_dot(np.eye(ny)[:, i]) for i in range(ny)]).T
        jac_x = np.array([self.proj_x.jac_dot(np.eye(nx)[:, i]) for i in range(nx)]).T
        jac_z_ref = np.load(os.path.join(self.path, 'jac_z.npy'))
        jac_y_ref = np.load(os.path.join(self.path, 'jac_y.npy'))
        jac_x_ref = np.load(os.path.join(self.path, 'jac_x.npy'))
        assert_allclose(jac_z, jac_z_ref,
                        err_msg='Unexpected behaviour in the the jacobi matrix! (z-axis)')
        assert_allclose(jac_y, jac_y_ref,
                        err_msg='Unexpected behaviour in the the jacobi matrix! (y-axis)')
        assert_allclose(jac_x, jac_x_ref,
                        err_msg='Unexpected behaviour in the the jacobi matrix! (x-axis)')

    def test_SimpleProjector_jac_T_dot(self):
        mz = self.proj_z.m
        my = self.proj_y.m
        mx = self.proj_x.m
        jac_T_z = np.array([self.proj_z.jac_T_dot(np.eye(mz)[:, i]) for i in range(mz)]).T
        jac_T_y = np.array([self.proj_y.jac_T_dot(np.eye(my)[:, i]) for i in range(my)]).T
        jac_T_x = np.array([self.proj_x.jac_T_dot(np.eye(mx)[:, i]) for i in range(mx)]).T
        jac_T_z_ref = np.load(os.path.join(self.path, 'jac_z.npy')).T
        jac_T_y_ref = np.load(os.path.join(self.path, 'jac_y.npy')).T
        jac_T_x_ref = np.load(os.path.join(self.path, 'jac_x.npy')).T
        assert_allclose(jac_T_z, jac_T_z_ref,
                        err_msg='Unexpected behaviour in the the transp. jacobi matrix! (z-axis)')
        assert_allclose(jac_T_y, jac_T_y_ref,
                        err_msg='Unexpected behaviour in the the transp. jacobi matrix! (y-axis)')
        assert_allclose(jac_T_x, jac_T_x_ref,
                        err_msg='Unexpected behaviour in the the transp. jacobi matrix! (x-axis)')


class TestCaseXTiltProjector(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_projector')
        self.magdata = load_vectordata(os.path.join(self.path, 'ref_magdata.hdf5'))
        self.proj_00 = XTiltProjector(self.magdata.dim, tilt=0)
        self.proj_45 = XTiltProjector(self.magdata.dim, tilt=pi / 4)
        self.proj_90 = XTiltProjector(self.magdata.dim, tilt=pi / 2)

    def tearDown(self):
        self.path = None
        self.magdata = None
        self.proj_00 = None
        self.proj_45 = None
        self.proj_90 = None

    def test_XTiltProjector_call(self):
        mag_proj_00 = self.proj_00(self.magdata)
        mag_proj_45 = self.proj_45(self.magdata)
        mag_proj_90 = self.proj_90(self.magdata)
        mag_proj_00_ref = load_vectordata(
            os.path.join(self.path, 'ref_mag_proj_x00.hdf5'))
        mag_proj_45_ref = load_vectordata(
            os.path.join(self.path, 'ref_mag_proj_x45.hdf5'))
        mag_proj_90_ref = load_vectordata(
            os.path.join(self.path, 'ref_mag_proj_x90.hdf5'))
        assert_allclose(mag_proj_00.field, mag_proj_00_ref.field,
                        err_msg='Unexpected behaviour in XTiltProjector (0°)')
        assert_allclose(mag_proj_45.field, mag_proj_45_ref.field,
                        err_msg='Unexpected behaviour in XTiltProjector (45°)')
        assert_allclose(mag_proj_90.field, mag_proj_90_ref.field,
                        err_msg='Unexpected behaviour in XTiltProjector (90°)')

    def test_XTiltProjector_jac_dot(self):
        mag_vec = self.magdata.field_vec
        mag_proj_00 = self.proj_00.jac_dot(mag_vec).reshape((2,) + self.proj_00.dim_uv)
        mag_proj_45 = self.proj_45.jac_dot(mag_vec).reshape((2,) + self.proj_45.dim_uv)
        mag_proj_90 = self.proj_90.jac_dot(mag_vec).reshape((2,) + self.proj_90.dim_uv)
        mag_proj_00_ref = load_vectordata(
            os.path.join(self.path, 'ref_mag_proj_x00.hdf5'))
        mag_proj_45_ref = load_vectordata(
            os.path.join(self.path, 'ref_mag_proj_x45.hdf5'))
        mag_proj_90_ref = load_vectordata(
            os.path.join(self.path, 'ref_mag_proj_x90.hdf5'))
        assert_allclose(mag_proj_00, mag_proj_00_ref.field[:2, 0, ...],
                        err_msg='Inconsistency between __call__() and jac_dot()! (0°)')
        assert_allclose(mag_proj_45, mag_proj_45_ref.field[:2, 0, ...],
                        err_msg='Inconsistency between __call__() and jac_dot()! (45°)')
        assert_allclose(mag_proj_90, mag_proj_90_ref.field[:2, 0, ...],
                        err_msg='Inconsistency between __call__() and jac_dot()! (90°)')
        n00 = self.proj_00.n
        n45 = self.proj_45.n
        n90 = self.proj_90.n
        jac_00 = np.array([self.proj_00.jac_dot(np.eye(n00)[:, i]) for i in range(n00)]).T
        jac_45 = np.array([self.proj_45.jac_dot(np.eye(n45)[:, i]) for i in range(n45)]).T
        jac_90 = np.array([self.proj_90.jac_dot(np.eye(n90)[:, i]) for i in range(n90)]).T
        jac_00_ref = np.load(os.path.join(self.path, 'jac_x00.npy'))
        jac_45_ref = np.load(os.path.join(self.path, 'jac_x45.npy'))
        jac_90_ref = np.load(os.path.join(self.path, 'jac_x90.npy'))
        assert_allclose(jac_00, jac_00_ref,
                        err_msg='Unexpected behaviour in the the jacobi matrix! (0°)')
        assert_allclose(jac_45, jac_45_ref,
                        err_msg='Unexpected behaviour in the the jacobi matrix! (45°)')
        assert_allclose(jac_90, jac_90_ref,
                        err_msg='Unexpected behaviour in the the jacobi matrix! (90°)')

    def test_XTiltProjector_jac_T_dot(self):
        m00 = self.proj_00.m
        m45 = self.proj_45.m
        m90 = self.proj_90.m
        jac_T_00 = np.array([self.proj_00.jac_T_dot(np.eye(m00)[:, i]) for i in range(m00)]).T
        jac_T_45 = np.array([self.proj_45.jac_T_dot(np.eye(m45)[:, i]) for i in range(m45)]).T
        jac_T_90 = np.array([self.proj_90.jac_T_dot(np.eye(m90)[:, i]) for i in range(m90)]).T
        jac_T_00_ref = np.load(os.path.join(self.path, 'jac_x00.npy')).T
        jac_T_45_ref = np.load(os.path.join(self.path, 'jac_x45.npy')).T
        jac_T_90_ref = np.load(os.path.join(self.path, 'jac_x90.npy')).T
        assert_allclose(jac_T_00, jac_T_00_ref,
                        err_msg='Unexpected behaviour in the the transp. jacobi matrix! (0°)')
        assert_allclose(jac_T_45, jac_T_45_ref,
                        err_msg='Unexpected behaviour in the the transp. jacobi matrix! (45°)')
        assert_allclose(jac_T_90, jac_T_90_ref,
                        err_msg='Unexpected behaviour in the the transp. jacobi matrix! (90°)')


class TestCaseYTiltProjector(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_projector')
        self.magdata = load_vectordata(os.path.join(self.path, 'ref_magdata.hdf5'))
        self.proj_00 = YTiltProjector(self.magdata.dim, tilt=0)
        self.proj_45 = YTiltProjector(self.magdata.dim, tilt=pi / 4)
        self.proj_90 = YTiltProjector(self.magdata.dim, tilt=pi / 2)

    def tearDown(self):
        self.path = None
        self.magdata = None
        self.proj_00 = None
        self.proj_45 = None
        self.proj_90 = None

    def test_XTiltProjector_call(self):
        mag_proj_00 = self.proj_00(self.magdata)
        mag_proj_45 = self.proj_45(self.magdata)
        mag_proj_90 = self.proj_90(self.magdata)
        mag_proj_00_ref = load_vectordata(
            os.path.join(self.path, 'ref_mag_proj_y00.hdf5'))
        mag_proj_45_ref = load_vectordata(
            os.path.join(self.path, 'ref_mag_proj_y45.hdf5'))
        mag_proj_90_ref = load_vectordata(
            os.path.join(self.path, 'ref_mag_proj_y90.hdf5'))
        assert_allclose(mag_proj_00.field, mag_proj_00_ref.field,
                        err_msg='Unexpected behaviour in XTiltProjector (0°)')
        assert_allclose(mag_proj_45.field, mag_proj_45_ref.field,
                        err_msg='Unexpected behaviour in XTiltProjector (45°)')
        assert_allclose(mag_proj_90.field, mag_proj_90_ref.field,
                        err_msg='Unexpected behaviour in XTiltProjector (90°)')

    def test_XTiltProjector_jac_dot(self):
        mag_vec = self.magdata.field_vec
        mag_proj_00 = self.proj_00.jac_dot(mag_vec).reshape((2,) + self.proj_00.dim_uv)
        mag_proj_45 = self.proj_45.jac_dot(mag_vec).reshape((2,) + self.proj_45.dim_uv)
        mag_proj_90 = self.proj_90.jac_dot(mag_vec).reshape((2,) + self.proj_90.dim_uv)
        mag_proj_00_ref = load_vectordata(
            os.path.join(self.path, 'ref_mag_proj_y00.hdf5'))
        mag_proj_45_ref = load_vectordata(
            os.path.join(self.path, 'ref_mag_proj_y45.hdf5'))
        mag_proj_90_ref = load_vectordata(
            os.path.join(self.path, 'ref_mag_proj_y90.hdf5'))
        assert_allclose(mag_proj_00, mag_proj_00_ref.field[:2, 0, ...],
                        err_msg='Inconsistency between __call__() and jac_dot()! (0°)')
        assert_allclose(mag_proj_45, mag_proj_45_ref.field[:2, 0, ...],
                        err_msg='Inconsistency between __call__() and jac_dot()! (45°)')
        assert_allclose(mag_proj_90, mag_proj_90_ref.field[:2, 0, ...],
                        err_msg='Inconsistency between __call__() and jac_dot()! (90°)')
        n00 = self.proj_00.n
        n45 = self.proj_45.n
        n90 = self.proj_90.n
        jac_00 = np.array([self.proj_00.jac_dot(np.eye(n00)[:, i]) for i in range(n00)]).T
        jac_45 = np.array([self.proj_45.jac_dot(np.eye(n45)[:, i]) for i in range(n45)]).T
        jac_90 = np.array([self.proj_90.jac_dot(np.eye(n90)[:, i]) for i in range(n90)]).T
        jac_00_ref = np.load(os.path.join(self.path, 'jac_y00.npy'))
        jac_45_ref = np.load(os.path.join(self.path, 'jac_y45.npy'))
        jac_90_ref = np.load(os.path.join(self.path, 'jac_y90.npy'))
        assert_allclose(jac_00, jac_00_ref,
                        err_msg='Unexpected behaviour in the the jacobi matrix! (0°)')
        assert_allclose(jac_45, jac_45_ref,
                        err_msg='Unexpected behaviour in the the jacobi matrix! (45°)')
        assert_allclose(jac_90, jac_90_ref,
                        err_msg='Unexpected behaviour in the the jacobi matrix! (90°)')

    def test_YTiltProjector_jac_T_dot(self):
        m00 = self.proj_00.m
        m45 = self.proj_45.m
        m90 = self.proj_90.m
        jac_T_00 = np.array([self.proj_00.jac_T_dot(np.eye(m00)[:, i]) for i in range(m00)]).T
        jac_T_45 = np.array([self.proj_45.jac_T_dot(np.eye(m45)[:, i]) for i in range(m45)]).T
        jac_T_90 = np.array([self.proj_90.jac_T_dot(np.eye(m90)[:, i]) for i in range(m90)]).T
        jac_T_00_ref = np.load(os.path.join(self.path, 'jac_y00.npy')).T
        jac_T_45_ref = np.load(os.path.join(self.path, 'jac_y45.npy')).T
        jac_T_90_ref = np.load(os.path.join(self.path, 'jac_y90.npy')).T
        assert_allclose(jac_T_00, jac_T_00_ref,
                        err_msg='Unexpected behaviour in the the transp. jacobi matrix! (0°)')
        assert_allclose(jac_T_45, jac_T_45_ref,
                        err_msg='Unexpected behaviour in the the transp. jacobi matrix! (45°)')
        assert_allclose(jac_T_90, jac_T_90_ref,
                        err_msg='Unexpected behaviour in the the transp. jacobi matrix! (90°)')


# TODO: Test RotTiltProjector!!!

if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)

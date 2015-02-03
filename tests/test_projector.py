# -*- coding: utf-8 -*-
"""Testcase for the projector module."""


import os
import unittest

from numpy import pi
from numpy.testing import assert_allclose

from pyramid.projector import XTiltProjector, YTiltProjector, SimpleProjector
from pyramid.magdata import MagData


class TestCaseProjector(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_projector')
        self.mag_data = MagData.load_from_netcdf4(os.path.join(self.path, 'ref_mag_data.nc'))

    def tearDown(self):
        self.path = None
        self.mag_data = None

    def test_simple_projector(self):
        mag_proj_z = SimpleProjector(self.mag_data.dim, axis='z')(self.mag_data)
        mag_proj_y = SimpleProjector(self.mag_data.dim, axis='y')(self.mag_data)
        mag_proj_x = SimpleProjector(self.mag_data.dim, axis='x')(self.mag_data)
        mag_proj_z_ref = MagData.load_from_netcdf4(os.path.join(self.path, 'ref_mag_proj_z.nc'))
        mag_proj_y_ref = MagData.load_from_netcdf4(os.path.join(self.path, 'ref_mag_proj_y.nc'))
        mag_proj_x_ref = MagData.load_from_netcdf4(os.path.join(self.path, 'ref_mag_proj_x.nc'))
        assert_allclose(mag_proj_z.magnitude, mag_proj_z_ref.magnitude,
                        err_msg='Unexpected behaviour in SimpleProjector (z-axis)')
        assert_allclose(mag_proj_y.magnitude, mag_proj_y_ref.magnitude,
                        err_msg='Unexpected behaviour in SimpleProjector (y-axis)')
        assert_allclose(mag_proj_x.magnitude, mag_proj_x_ref.magnitude,
                        err_msg='Unexpected behaviour in SimpleProjector (x-axis)')

    def test_x_tilt_projector(self):
        mag_proj_00 = XTiltProjector(self.mag_data.dim, tilt=0)(self.mag_data)
        mag_proj_45 = XTiltProjector(self.mag_data.dim, tilt=pi/4)(self.mag_data)
        mag_proj_90 = XTiltProjector(self.mag_data.dim, tilt=pi/2)(self.mag_data)
        mag_proj_00_ref = MagData.load_from_netcdf4(os.path.join(self.path, 'ref_mag_proj_x00.nc'))
        mag_proj_45_ref = MagData.load_from_netcdf4(os.path.join(self.path, 'ref_mag_proj_x45.nc'))
        mag_proj_90_ref = MagData.load_from_netcdf4(os.path.join(self.path, 'ref_mag_proj_x90.nc'))
        assert_allclose(mag_proj_00.magnitude, mag_proj_00_ref.magnitude,
                        err_msg='Unexpected behaviour in XTiltProjector (0°)')
        assert_allclose(mag_proj_45.magnitude, mag_proj_45_ref.magnitude,
                        err_msg='Unexpected behaviour in XTiltProjector (45°)')
        assert_allclose(mag_proj_90.magnitude, mag_proj_90_ref.magnitude,
                        err_msg='Unexpected behaviour in XTiltProjector (90°)')

    def test_y_tilt_projector(self):
        mag_proj_00 = YTiltProjector(self.mag_data.dim, tilt=0)(self.mag_data)
        mag_proj_45 = YTiltProjector(self.mag_data.dim, tilt=pi/4)(self.mag_data)
        mag_proj_90 = YTiltProjector(self.mag_data.dim, tilt=pi/2)(self.mag_data)
        mag_proj_00_ref = MagData.load_from_netcdf4(os.path.join(self.path, 'ref_mag_proj_y00.nc'))
        mag_proj_45_ref = MagData.load_from_netcdf4(os.path.join(self.path, 'ref_mag_proj_y45.nc'))
        mag_proj_90_ref = MagData.load_from_netcdf4(os.path.join(self.path, 'ref_mag_proj_y90.nc'))
        assert_allclose(mag_proj_00.magnitude, mag_proj_00_ref.magnitude,
                        err_msg='Unexpected behaviour in YTiltProjector (0°)')
        assert_allclose(mag_proj_45.magnitude, mag_proj_45_ref.magnitude,
                        err_msg='Unexpected behaviour in YTiltProjector (45°)')
        assert_allclose(mag_proj_90.magnitude, mag_proj_90_ref.magnitude,
                        err_msg='Unexpected behaviour in YTiltProjector (90°)')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseProjector)
    unittest.TextTestRunner(verbosity=2).run(suite)

# -*- coding: utf-8 -*-
"""Testcase for the holoimage module."""


import os
import unittest
import numpy as np
from numpy import pi

from pyramid.phasemap import PhaseMap
import pyramid.holoimage as hi


class TestCaseHoloImage(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_holoimage')
        phase = np.zeros((4, 4))
        phase[1:-1, 1:-1] = pi/4
        self.phase_map = PhaseMap(10.0, phase)

    def tearDown(self):
        self.path = None
        self.phase_map = None

    def test_holo_image(self):
        img = hi.holo_image(self.phase_map)
        arr = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
        holo_img_r, holo_img_g, holo_img_b = arr[..., 0], arr[..., 1], arr[..., 2]
        ref_holo_img_r = np.loadtxt(os.path.join(self.path, 'ref_holo_img_r.txt'))
        ref_holo_img_g = np.loadtxt(os.path.join(self.path, 'ref_holo_img_g.txt'))
        ref_holo_img_b = np.loadtxt(os.path.join(self.path, 'ref_holo_img_b.txt'))
        hi.display(img)
        np.testing.assert_equal(holo_img_r, ref_holo_img_r,
                                'Unexpected behavior in holo_image() (r-component)!')
        np.testing.assert_equal(holo_img_g, ref_holo_img_g,
                                'Unexpected behavior in holo_image() (g-component)!')
        np.testing.assert_equal(holo_img_b, ref_holo_img_b,
                                'Unexpected behavior in holo_image() (b-component)!')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseHoloImage)
    unittest.TextTestRunner(verbosity=2).run(suite)

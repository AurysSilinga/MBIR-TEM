# -*- coding: utf-8 -*-
"""Testcase for the holoimage module."""


import unittest
import pyramid.holoimage as hi


class TestCaseHoloImage(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_holo_image(self):
        pass

    def test_make_color_wheel(self):
        pass

    def test_display(self):
        pass

    def test_display_combined(self):
        pass

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseHoloImage)
    unittest.TextTestRunner(verbosity=2).run(suite)

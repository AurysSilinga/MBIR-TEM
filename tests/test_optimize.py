# -*- coding: utf-8 -*-
"""Testcase for the reconstructor module."""


import os
import unittest

import pyramid.reconstructor as rc


class TestCaseReconstructor(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_phasemapper')

    def tearDown(self):
        self.path = None

    def test_reconstruct_simple_leastsq(self):
        raise AssertionError


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseReconstructor)
    unittest.TextTestRunner(verbosity=2).run(suite)

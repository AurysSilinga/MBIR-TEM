# -*- coding: utf-8 -*-
"""Testcase for the reconstructor module."""


import unittest
import pyramid.reconstructor as rc


class TestCaseReconstructor(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_template(self):
        pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseReconstructor)
    unittest.TextTestRunner(verbosity=2).run(suite)

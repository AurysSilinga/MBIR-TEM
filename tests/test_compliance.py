# -*- coding: utf-8 -*-
"""Testcase for the magcreator module."""


import os
import sys
import datetime
import unittest

import pep8


class TestCaseCompliance(unittest.TestCase):
    """Class for checking compliance of pyramid."""  # TODO: Docstring

    def setUp(self):
        self.path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]  # Pyramid dir

    def tearDown(self):
        self.path = None

    def get_files_to_check(self, rootdir):
        filepaths = []
        for root, dirs, files in os.walk(rootdir):
            for filename in files:
                if ((filename.endswith('.py') or filename.endswith('.pyx'))
                and root != os.path.join(self.path, 'scripts', 'gui')):
                    filepaths.append(os.path.join(root, filename))
        return filepaths

    def test_pep8(self):
        # TODO: Docstring
        files = self.get_files_to_check(os.path.join(self.path, 'pyramid')) \
            + self.get_files_to_check(os.path.join(self.path, 'scripts')) \
            + self.get_files_to_check(os.path.join(self.path, 'tests'))
        ignores = ('E226', 'E128')
        pep8.MAX_LINE_LENGTH = 99
        pep8style = pep8.StyleGuide(quiet=False)
        pep8style.options.ignore = ignores

        stdout_buffer = sys.stdout
        with open(os.path.join(self.path, 'output', 'pep8_log.txt'), 'w') as sys.stdout:
            print '<<< PEP8 LOGFILE >>>'
            print 'RUN:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print 'IGNORED RULES:', ', '.join(ignores)
            print 'MAX LINE LENGTH:', pep8.MAX_LINE_LENGTH
            print '\nERRORS AND WARNINGS:'
            result = pep8style.check_files(files)
            if result.total_errors == 0:
                print 'No Warnings or Errors detected!'
            else:
                print '\n{} Warnings and Errors detected!'.format(result.total_errors)
        sys.stdout = stdout_buffer

        error_message = 'Found %s Errors and Warnings!' % result.total_errors
        self.assertEqual(result.total_errors, 0, error_message)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseCompliance)
    unittest.TextTestRunner(verbosity=2).run(suite)

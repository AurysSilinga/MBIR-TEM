# -*- coding: utf-8 -*-
"""Testcase for the magcreator module."""


import os
import sys
import datetime
import unittest

import re

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
        ignores = ('E125', 'E226', 'E228')
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
                print '---->   {} Warnings and Errors detected!'.format(result.total_errors)
            print '\nTODOS:'
            todos_found = False
            todo_count = 0
            regex = ur'# TODO: (.*)'
            for py_file in files:
                with open(py_file) as f:
                    for line in f:
                        todo = re.findall(regex, line)
                        if todo and not todo[0] == "(.*)'":
                            todos_found = True
                            todo_count += 1
                            print '{}: {}'.format(f.name, todo[0])
            if todos_found:
                print '---->   {} TODOs found!'.format(todo_count)
            else:
                print 'No TODOS found!'

        sys.stdout = stdout_buffer

        error_message = 'Found %s Errors and Warnings!' % result.total_errors
        self.assertEqual(result.total_errors, 0, error_message)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseCompliance)
    unittest.TextTestRunner(verbosity=2).run(suite)

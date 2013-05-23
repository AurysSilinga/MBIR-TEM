# -*- coding: utf-8 -*-
"""Testcase for the magcreator module."""


import os
import unittest


class TestCaseCompliance(unittest.TestCase):
    """
    Class for checking compliance of pyjurassic.
    """

    def setUp(self):
        # self.getPaths()
        # self.data_dir = "test_jrp_data"
        # self.src_ctl = "ctl.py"
        # remove E201 to get a working version of str(- b)
        # remove E203 to get a working version of a[:, :]
        # remove E501 as 80 chars are not enough
        self.ignores = ["E201", "E203", "E501"]
        self.options = ["dummy_name", "--ignore", ",".join(self.ignores), "--repeat"]
        self.options += ["--exclude", "collections_python27.py"]
        try:
            import pep8
            self.pep8 = pep8
        except ImportError:
            self.pep8 = None
            print("\nWARNING: You do not have package pep8!")

    def countErrors(self):
        """
        Counts the relevant errors from the pep8 counter structure.
        """
        return sum([self.pep8.options.counters[key]
                    for key in self.pep8.options.messages.keys()
                    if key not in self.ignores])

    def checkDirectory(self, dir_name):
        """
        Checks all python files in the supplied directory and subdirectories.
        """
        self.pep8.process_options(self.options)
        self.pep8.input_dir(dir_name)
        return self.countErrors()

    def test_pep8(self):
        """
        Tests all directories containing python files for PEP8 compliance.
        """
        self.assertTrue(self.pep8 is not None,
                        msg="Install Python pep8 module to fully execute testbench!")
        errors = 0
        for dir_name in ["test", os.environ["BUILDDIR"]]:
            errors += self.checkDirectory(dir_name)
        self.assertEqual(errors, 0)

#    def test_variables(self):
#        """
#        Function for checking that all attributes are present.
#        """
#        try:
#            import gloripy
##            for name in ['X', 'Y', 'B',
##                         'APR_VAL', 'APR_STD', 'FINAL_VAL', 'FINAL_STD',
##                         'IG_VAL', 'TRUE_VAL',
##                         'CARTESIAN', 'GEO', 'SPHERICAL',
##                         'P', 'T', 'C']:
##                self.assertTrue(hasattr(j, name), msg=str(name) + " is missing.")
#        except ImportError:
#            self.assertTrue(False, msg="could not import gloripy")

#    def test_import(self):
#        """
#        Checks that all modules are importable.
#        """
#        module_list = ["gloripy",
#                       "gloripy.pylevel0",
#                      ]
#        for module in module_list:
#            try:
#                exec("import " + module)
#                importable = True
#            except ImportError:
#                importable = False
#            self.assertTrue(importable, msg="importing " + module + " failed")

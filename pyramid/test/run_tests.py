# -*- coding: utf-8 -*-
"""
Unittests for pyramid.
"""

import unittest
import sys
from test_dataloader import *
from test_compliance import *


def run():
    
    suite  = unittest.TestSuite()
    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=2)
    
    suite.addTest(loader.loadTestsFromTestCase(TestCaseDataloader))
    suite.addTest(loader.loadTestsFromTestCase(TestCaseCompliance))
    runner.run(suite)
    
    #TODO: why that?
#    result = runner.run(suite)
#    if result.wasSuccessful():
#        sys.exit(0)
#    else:
#        sys.exit(1)


if __name__ == '__main__':
    run()
    
    
    
    

#if __name__ == '__main__':
#    suite = unittest.TestLoader().loadTestsFromTestCase(TestSuite)
#    unittest.TextTestRunner(verbosity=2).run(suite)



''' GLORIPY VERSION '''
##! /usr/bin/python
#"""
#Unittests for pyjurassic.
#"""
#
#import unittest
#import os
#import sys
#import tests
#
#
#def main():
#    all_tests = unittest.TestSuite()
#    tl = unittest.defaultTestLoader
#    if os.getenv("BOOST_TESTS_TO_RUN") is not None:
#        return
#    elif os.getenv("PYTHON_TESTS_TO_RUN") is None:
#        all_tests.addTest(tl.loadTestsFromName("tests"))
#    else:
#        all_tests.addTest(tl.loadTestsFromName(os.getenv("PYTHON_TESTS_TO_RUN")))
#
#    runner = unittest.TextTestRunner(verbosity=2)
#    res = runner.run(all_tests)
#    if res.wasSuccessful():
#        sys.exit(0)
#    else:
#        sys.exit(1)
#
#
#if __name__ == '__main__':
#    main()
import unittest
from test_compliance import TestCaseCompliance
from test_magcreator import TestCaseMagCreator
from test_magdata import TestCaseMagData
from test_projector import TestCaseProjector
from test_phasemapper import TestCasePhaseMapper
from test_phasemap import TestCasePhaseMap
from test_holoimage import TestCaseHoloImage
from test_analytic import TestCaseAnalytic
from test_reconstructor import TestCaseReconstructor


def load_suite():
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(TestCaseCompliance))
    suite.addTest(loader.loadTestsFromTestCase(TestCaseMagCreator))
    suite.addTest(loader.loadTestsFromTestCase(TestCaseMagData))
    suite.addTest(loader.loadTestsFromTestCase(TestCaseProjector))
    suite.addTest(loader.loadTestsFromTestCase(TestCasePhaseMapper))
    suite.addTest(loader.loadTestsFromTestCase(TestCasePhaseMap))
    suite.addTest(loader.loadTestsFromTestCase(TestCaseHoloImage))
    suite.addTest(loader.loadTestsFromTestCase(TestCaseAnalytic))
    suite.addTest(loader.loadTestsFromTestCase(TestCaseReconstructor))
    return suite

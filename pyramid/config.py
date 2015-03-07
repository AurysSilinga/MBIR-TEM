# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 10:59:42 2015

@author: Jan
"""

import os


__all__ = ['DIR_PACKAGE', 'DIR_FILES', 'LOGGING_CONFIG']


DIR_PACKAGE = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DIR_FILES = os.path.abspath(os.path.join(DIR_PACKAGE, os.pardir, 'files'))
LOGGING_CONFIG = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logging.ini')

del os

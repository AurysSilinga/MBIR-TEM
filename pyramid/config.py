# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 10:59:42 2015

@author: Jan
"""

import os


__all__ = ['DIR_PACKAGE', 'DIR_MAGDATA', 'DIR_PHASEMAP']


DIR_PACKAGE = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DIR_MAGDATA = os.path.abspath(os.path.join(DIR_PACKAGE, os.pardir, 'files', 'magdata'))
DIR_PHASEMAP = os.path.abspath(os.path.join(DIR_PACKAGE, os.pardir, 'files', 'phasemap'))

del os

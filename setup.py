# -*- coding: utf-8 -*-
"""
Created on Fri May 03 10:27:04 2013

@author: Jan
"""

#call with: python setup.py build_ext --inplace --compiler=mingw32

import os
import glob
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'Pyramid',
    version = '0.1',    
    description = 'PYthon based Reconstruction Algorithm for MagnetIc Distributions',
    author = 'Jan Caron',
    author_email = 'j.caron@fz-juelich.de',
    packages = ['pyramid'],
    ext_modules = cythonize(glob.glob(os.path.join('pyramid','numcore','*.pyx')))
)

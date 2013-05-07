# -*- coding: utf-8 -*-
"""
Created on Fri May 03 10:27:04 2013

@author: Jan
"""

#call with: python setup.py build_ext --inplace --compiler=mingw32

import glob
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'Pyramex',
    version = '0.1',    
    description = 'Pyramid Cython Extensions',
    author = 'Jan Caron',
    author_email = 'j.caron@fz-juelich.de',
    ext_modules = cythonize(glob.glob('*.pyx'))
)
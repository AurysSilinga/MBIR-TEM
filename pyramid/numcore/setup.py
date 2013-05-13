"""
Created on Fri May 03 10:27:04 2013

@author: Jan
"""

#call with: python setup.py build_ext --inplace --compiler=mingw32

import glob
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

#setup(
#    cmdclass = {'build_ext': build_ext},
#    ext_modules = [Extension("multiply",
#                             sources=["multiply.pyx", "c_multiply.c"],
#                             include_dirs=[numpy.get_include()])],
#)

setup(
    name = 'Pyramex',
    version = '0.1',    
    description = 'Pyramid Cython Extensions',
    author = 'Jan Caron',
    author_email = 'j.caron@fz-juelich.de',
    ext_modules = cythonize(glob.glob('*.pyx'))
)

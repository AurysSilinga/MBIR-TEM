# -*- coding: utf-8 -*-
"""
Created on Fri May 03 10:27:04 2013

@author: Jan
"""


# Build extensions: 'python setup.py build_ext -i clean'
# Install package:  'python setup.py install clean'


import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


setup(

      name = 'Pyramid',
      version = '0.1',
      description = 'PYthon based Reconstruction Algorithm for MagnetIc Distributions',
      author = 'Jan Caron',
      author_email = 'j.caron@fz-juelich.de',
      
      packages = ['pyramid', 'pyramid.numcore'],
      include_dirs = [numpy.get_include()],
                      
      cmdclass = {'build_ext': build_ext},              
      ext_package = 'pyramid/numcore',
      ext_modules = [
          Extension('phase_mag_real', ['pyramid/numcore/phase_mag_real.pyx'], 
                    include_dirs = [numpy.get_include(), numpy.get_numarray_include()],
                    extra_compile_args=["-march=pentium", "-mtune=pentium"]
                    )
          ]

)

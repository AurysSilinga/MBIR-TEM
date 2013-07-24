# -*- coding: utf-8 -*-
"""
Created on Fri May 03 10:27:04 2013

@author: Jan
"""


# Build extensions: 'python setup.py build_ext -i clean'
# Install package:  'python setup.py install clean'


import numpy

import os
import sys
import sysconfig
#import glob

#from distutils.core import setup
from distutils.command.build import build
#from distutils.extension import Extension

from Cython.Distutils import build_ext

from setuptools import setup, find_packages

from setuptools.extension import Extension


def make_hgrevision(target, source, env):
    import subprocess as sp
    output = sp.Popen(["hg", "id", "-i"], stdout=sp.PIPE).communicate()[0]
    hgrevision_cc = file(str(target[0]), "w")
    hgrevision_cc.write('HG_Revision = "{0}"\n'.format(output.strip()))
    hgrevision_cc.close()

print '\n------------------------------------------------------------------------------'

setup(
      name = 'Pyramid',
      version = '0.1',
      description = 'PYthon based Reconstruction Algorithm for MagnetIc Distributions',
      author = 'Jan Caron',
      author_email = 'j.caron@fz-juelich.de',
      
      packages = find_packages(exclude=['test']),#['pyramid', 'pyramid.numcore', 'test', 'scripts'],
      include_dirs = [numpy.get_include()],
      requires = ['numpy', 'matplotlib'],
      
      scripts = ['scripts/create_logo.py'],
      test_suite = 'test',
      
      cmdclass = {'build_ext': build_ext},
      ext_package = 'pyramid/numcore',
      ext_modules = [
          Extension('phase_mag_real', ['pyramid/numcore/phase_mag_real.pyx'], 
                    include_dirs = [numpy.get_include(), numpy.get_numarray_include()],
                    extra_compile_args=["-march=native", "-mtune=native"]
                    )
          ]
)

print '------------------------------------------------------------------------------\n'

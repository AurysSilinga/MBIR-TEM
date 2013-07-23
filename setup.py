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

from distutils.core import setup
from distutils.command.build import build
from distutils.extension import Extension
from Cython.Distutils import build_ext

class custom_build(build):
    def run(self):
        build.run(self)
        print 'Test'

def distutils_dir_name(dname):
    '''Returns the name of a distutils build directory'''
    path = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return path.format(dirname=dname, platform=sysconfig.get_platform(), version=sys.version_info)


print '\n------------------------------------------------------------------------------'

build_path = os.path.join('build', distutils_dir_name('lib'))

setup(

      name = 'Pyramid',
      version = '0.1',
      description = 'PYthon based Reconstruction Algorithm for MagnetIc Distributions',
      author = 'Jan Caron',
      author_email = 'j.caron@fz-juelich.de',
      
      packages = ['pyramid', 'pyramid.numcore', 'pyramid.test'],
      include_dirs = [numpy.get_include()],
      requires = ['numpy', 'matplotlib'],
                      
      cmdclass = {'build_ext': build_ext, 'build': custom_build},
      ext_package = 'pyramid/numcore',
      ext_modules = [
          Extension('phase_mag_real', ['pyramid/numcore/phase_mag_real.pyx'], 
                    include_dirs = [numpy.get_include(), numpy.get_numarray_include()],
                    extra_compile_args=["-march=native", "-mtune=native"]
                    )
          ]

)

import os
print os.getcwd()

print '------------------------------------------------------------------------------\n'

#import pyramid.test as test
#test.run_tests()

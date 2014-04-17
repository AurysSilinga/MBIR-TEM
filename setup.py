# -*- coding: utf-8 -*-
"""Setup for testing, building, distributing and installing the 'Pyramid'-package"""


import numpy
import os
import sys
import sysconfig
import subprocess
from distutils.command.build import build
from Cython.Distutils import build_ext
from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension

from pyramid._version import __version__


class custom_build(build):
    '''Custom build command'''

    def make_hgrevision(self, target):
        output = subprocess.Popen(["hg", "id", "-i"], stdout=subprocess.PIPE).communicate()[0]
        hgrevision_cc = file(str(target), "w")
        hgrevision_cc.write('hg_revision = "{0}"'.format(output.strip()))
        hgrevision_cc.close()

    def run(self):
        build.run(self)
        print 'creating hg_revision.txt'
        self.make_hgrevision(os.path.join('build', get_build_path('lib'), 'hg_revision.txt'))


def get_build_path(dname):
    '''Returns the name of a distutils build directory'''
    path = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return path.format(dirname=dname, platform=sysconfig.get_platform(), version=sys.version_info)


def get_files(rootdir):
    '''Returns a list of .py-files inside rootdir'''
    filepaths = []
    for root, dirs, files in os.walk(rootdir):
        for filename in files:
            if filename.endswith('.py'):
                filepaths.append(os.path.join(root, filename))
    return filepaths


print '\n-------------------------------------------------------------------------------'

setup(
      name = 'Pyramid',
      version = __version__,
      description = 'PYthon based Reconstruction Algorithm for MagnetIc Distributions',
      author = 'Jan Caron',
      author_email = 'j.caron@fz-juelich.de',
      url = 'fz-juelich.de',

      packages = find_packages(exclude=['tests']),
      include_dirs = [numpy.get_include()],
      requires = ['numpy', 'matplotlib', 'mayavi'],

      scripts = get_files('scripts'),

      test_suite = 'tests',

      cmdclass = {'build_ext': build_ext, 'build': custom_build},

      ext_package = 'pyramid/numcore',
      ext_modules = [
          Extension('kernel_core', ['pyramid/numcore/kernel_core.pyx'],
                    include_dirs = [numpy.get_include(), numpy.get_numarray_include()],
                    extra_compile_args=['-march=native', '-mtune=native']
                    ),
          Extension('phasemapper_core', ['pyramid/numcore/phasemapper_core.pyx'],
                    include_dirs = [numpy.get_include(), numpy.get_numarray_include()],
                    extra_compile_args=['-march=native', '-mtune=native']
                    )
          ]
)

print '-------------------------------------------------------------------------------\n'

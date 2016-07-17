#!python
# coding=utf-8
"""Setup for testing, building, distributing and installing the 'Pyramid'-package"""

import os
import re
import subprocess
import sys
from distutils.command.build import build

import numpy
from Cython.Distutils import build_ext
from setuptools import setup, find_packages


DISTNAME = 'pyramid'
DESCRIPTION = 'PYthon based Reconstruction Algorithm for MagnetIc Distributions'
MAINTAINER = 'Jan Caron'
MAINTAINER_EMAIL = 'j.caron@fz-juelich.de'
URL = ''
VERSION = '0.1.0-dev'
PYTHON_VERSION = (2, 7)
DEPENDENCIES = {'numpy': (1, 10), 'cython': (0, 23)}
LONG_DESCRIPTION = 'long description (TODO!)'  # TODO: Long description!


def get_package_version(package):
    """Return the package version of the specified package.

    Parameters
    ----------
    package: basestring
        Name of the package whic should be checked.

    Returns
    -------
    version: tuple (N=3)
        Version number as a tuple.

    """
    version = []
    for version_attr in ('version', 'VERSION', '__version__'):
        if (hasattr(package, version_attr) and
                isinstance(getattr(package, version_attr), str)):
            version_info = getattr(package, version_attr, '')
            for part in re.split('\D+', version_info):
                try:
                    version.append(int(part))
                except ValueError:
                    pass
    return tuple(version)


def check_requirements():
    """Checks the requirements of the Pyramid package."""
    if sys.version_info < PYTHON_VERSION:
        raise SystemExit('You need Python version %d.%d or later.'
                         % PYTHON_VERSION)
    for package_name, min_version in DEPENDENCIES.items():
        dep_error = False
        try:
            package = __import__(package_name)
        except ImportError:
            dep_error = True
        else:
            package_version = get_package_version(package)
            if min_version > package_version:
                dep_error = True
        if dep_error:
            raise ImportError('You need `%s` version %d.%d or later.'
                              % ((package_name,) + min_version))


def hg_version():
    """Get the Mercurial reference identifier.

    Returns
    -------
    hg_ref: basestring
        The Mercurial reference identifier.

    """
    try:
        hg_rev = subprocess.check_output(['hg', 'id', '--id']).strip()
    except:
        hg_rev = "???"
    return hg_rev


def write_version_py(filename='pyramid/version.py'):
    """Write the version.py file.

    Parameters
    ----------
    filename: basestring, optional
        Write the version and hg_revision into the specified python file.
        Defaults to 'pyramid/version.py'.

    """
    version_string = '# -*- coding: utf-8 -*-\n' + \
                     '""""This file is generated automatically by the Pyramid `setup.py`"""\n' + \
                     'version = "{}"\n'.format(VERSION) + \
                     'hg_revision = "{}"\n'.format(hg_version())
    with open(os.path.join(os.path.dirname(__file__), filename), 'w') as vfile:
        vfile.write(version_string)


def get_files(rootdir):
    """Returns a list of .py-files inside rootdir.

    Parameters
    ----------
    rootdir: basestring
        Root directory in which to search for ``.py``-files.

    Returns
    -------
    filepaths: list
        List of filepaths which were found.

    """
    filepaths = []
    for root, dirs, files in os.walk(rootdir):
        for filename in files:
            if filename.endswith('.py'):
                filepaths.append(os.path.join(root, filename))
    return filepaths


print('\n-------------------------------------------------------------------------------')
print('checking requirements')
check_requirements()
print('write version.py')
write_version_py()
setup(name=DISTNAME,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=URL,
      version=VERSION,
      packages=find_packages(exclude=['tests']),
      include_dirs=[numpy.get_include()],
      requires=['numpy', 'scipy', 'matplotlib', 'Pillow',
                'mayavi', 'pyfftw', 'hyperspy', 'nose'],
      test_suite='nose.collector',
      cmdclass={'build_ext': build_ext, 'build': build})
print('-------------------------------------------------------------------------------\n')

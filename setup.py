#!python
# coding=utf-8
"""Setup for testing, building, distributing and installing the 'Pyramid'-package"""

import os
import re
import subprocess
import sys
import itertools
#from distutils.command.build import build

#import numpy
from setuptools import setup, find_packages


DISTNAME = 'pyramid'
DESCRIPTION = 'PYthon based Reconstruction Algorithm for MagnetIc Distributions'
MAINTAINER = 'Jan Caron'
MAINTAINER_EMAIL = 'j.caron@fz-juelich.de'
URL = ''
VERSION = '0.1.0.dev0'  # TODO: Better way?
PYTHON_VERSION = (2, 7)  # TODO: get rid of!!!
DEPENDENCIES = {'numpy': (1, 10)}  # TODO: get rid of!!!
LONG_DESCRIPTION = 'long description (TODO!)'  # TODO: Long description! put in (Readme?) file!


# TODO: get rid of superfluous functions!
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


def hg_version():  # TODO: Replace with GIT! Also check build output on GitLab! See numpy setup.py!
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


# TODO: Outsource stuff to setup.cfg? See https://github.com/pypa/setuptools/pull/862

# TODO: Use requirements.txt? extras_require for optional stuff (hyperspy, plotting)?
# TODO: After split of Pyramid, comment out and see what really is used (is e.g. scipy?)!
install_requires = ['numpy>=1.6', 'tqdm', 'scipy', 'matplotlib', 'Pillow', 'h5py',
                    'hyperspy', 'jutil', 'cmocean']

# TODO: extend extras_require for plotting and IO:
# TODO: extra: 'pyfftw', 'mayavi' (not easy to install... find a way!)
# TODO: See https://stackoverflow.com/a/28842733 for extras_require...
# TODO: ...replace [dev] with [IO] (hyperspy) and [plotting] (separate plotting library)!
extras_require = {
    # TODO: Test all if really needed! don't use nose, if possible (pure pytest)!
    'tests': ['pytest', 'pytest-runner', 'pytest-cov', 'pytest-flake8', 'coverage'],
    '3Dplot': ['qt==4.8', 'mayavi==4.5']
    # TODO: more for mayavi (plotting in general) and hyperspy, etc (see below)...
}

extras_require["all"] = list(itertools.chain(*list(extras_require.values())))


print('\n-------------------------------------------------------------------------------')
# print('checking requirements')  # TODO: Get rid of!
# check_requirements()
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
      packages=find_packages(exclude=['tests', 'doc']),  # TODO: necessary?
      #include_dirs=[numpy.get_include()],  # TODO: Maybe used for sphinx?!
      #setup_requires=['numpy>=1.6', 'pytest', 'pytest-runner'],
      #tests_require=['pytest', 'pytest-cov', 'pytest-flake8'],
      install_requires=install_requires,
      extras_require=extras_require,
      #cmdclass={'build': build}  # TODO: necessary?
    )
print('-------------------------------------------------------------------------------\n')

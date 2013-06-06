# -*- coding: utf-8 -*-
"""Script to initialize Cythons pyximport to ensure compatibility with MinGW compiler and NumPy."""


import os
import numpy
import pyximport


if os.name == 'nt':
    if 'CPATH' in os.environ:
        os.environ['CPATH'] = os.environ['CPATH'] + numpy.get_include()
    else:
        os.environ['CPATH'] = numpy.get_include()
#    # XXX: assuming that MinGW is installed in C:\MinGW (default)
#    #      for PythonXY the default is C:\MinGW-xy
#    if os.environ.has_key('PATH'):
#        os.environ['PATH'] = os.environ['PATH'] + ';C:\MinGW\bin'
#    else:
#        os.environ['PATH'] = 'C:\MinGW\bin'

    mingw_setup_args = {'options': {'build_ext': {'compiler': 'mingw32'}}}
    pyximport.install(setup_args=mingw_setup_args, inplace=True)


elif os.name == 'posix':
    if 'CFLAGS' in os.environ:
        os.environ['CFLAGS'] = os.environ['CFLAGS'] + ' -I' + numpy.get_include()
    else:
        os.environ['CFLAGS'] = ' -I' + numpy.get_include()

    pyximport.install(inplace=True)

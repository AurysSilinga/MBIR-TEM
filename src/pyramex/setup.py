# -*- coding: utf-8 -*-
"""
Created on Fri May 03 10:27:04 2013

@author: Jan
"""

#from distutils.core import setup, Extension
#
#module1 = Extension('c1',
#                    sources = ['c1.pyx'])
#
#setup (name = 'pyramex',
#       version = '1.0',
#       description = 'This is a demo package',
#       ext_modules = [module1])



from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

def get_extensions():
	return [
		Extension('hello', ['hello.pyx']),
		Extension('c1',    ['c1.pyx']),
		Extension('c2',    ['c2.pyx']),
		Extension('c3',    ['c3.pyx'])
		]

setup(
  name = 'pyramex',
  cmdclass = {'build_ext': build_ext},
  ext_modules = get_extensions()
)
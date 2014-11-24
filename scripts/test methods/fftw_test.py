# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:28:34 2014

@author: Jan
"""


import os

import numpy as np
import pyfftw

import pyramid
from pyramid import *
from jutil.taketime import TakeTime

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)

n = 2**20
print n

a = pyfftw.n_byte_align_empty(n, pyfftw.simd_alignment, 'complex128')
a[:] = np.random.randn(n) + 1j*np.random.randn(n)
pyfftw.interfaces.cache.enable()

with TakeTime('FFTW uncached'):
    b = pyfftw.interfaces.numpy_fft.fft(a)
with TakeTime('FFTW cached  '):
    b = pyfftw.interfaces.numpy_fft.fft(a)
with TakeTime('Numpy        '):
    c = np.fft.fft(a)
d = pyfftw.n_byte_align_empty(n, pyfftw.simd_alignment, 'complex128')
fft_object = pyfftw.FFTW(a, d)
with TakeTime('FFTW-object  '):
    d = fft_object(a, d)
fft_object = pyfftw.builders.fft(a, threads=1)
with TakeTime('FFTW-build 1t'):
    for i in range(100):
        d = fft_object(a, d)
fft_object = pyfftw.builders.fft(a, threads=4)
with TakeTime('FFTW-build 4t'):
    for i in range(100):
        d = fft_object(a, d)
fft_object = pyfftw.builders.fft(a, threads=8)
with TakeTime('FFTW-build 8t'):
    for i in range(100):
        d = fft_object(a, d)

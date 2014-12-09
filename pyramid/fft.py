# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 15:30:10 2014

@author: Jan
"""

# TODO: Document!

import numpy as np
import cPickle as pickle
import os

# pyFFTW depends on this
try:
    from collections import Counter  #analysis:ignore
except ImportError:
    import collections_python27
    import collections
    collections.Counter = collections_python27.Counter

try:
    import pyfftw
    BACKEND = 'fftw'
except ImportError:
    pyfftw = None
    BACKEND = 'numpy'
    print("pyFFTW module not found. Using numpy implementation.")


__all__ = ['PLANS', 'FLOAT', 'COMPLEX', 'NTHREADS', 'dump_wisdom', 'load_wisdom', 'zeros', 'empty',
           'configure_backend', 'fftn', 'ifftn', 'rfftn', 'irfftn', 'rfftn_adj', 'irfftn_adj']


class FFTWCache(object):

    def __init__(self):
        self.cache = dict()

    def add_fftw(self, fft_type, fftw_obj, s, axes, nthreads):
        in_arr = fftw_obj.get_input_array()
        key = (fft_type, in_arr.shape, in_arr.dtype, nthreads)
        self.cache[key] = fftw_obj

    def lookup_fftw(self, fft_type, in_arr, s, axes, nthreads):
        key = (fft_type, in_arr.shape, in_arr.dtype, nthreads)
        return self.cache.get(key, None)

    def clear_cache(self):
        self.cache = dict()


PLANS = FFTWCache()
FLOAT = np.float32      # One convenient place to
COMPLEX = np.complex64  # change from 32 to 64 bit
NTHREADS = 1


# Numpy functions:
def _fftn_numpy(a, s=None, axes=None):
    return np.fft.fftn(a, s, axes)


def _ifftn_numpy(a, s=None, axes=None):
    return np.fft.ifftn(a, s, axes)


def _rfftn_numpy(a, s=None, axes=None):
    return np.fft.rfftn(a, s, axes)


def _irfftn_numpy(a, s=None, axes=None):
    return np.fft.irfftn(a, s, axes)


def _rfftn_adj_numpy(a):
    n = 2 * (a.shape[-1] - 1)
    out_shape = a.shape[:-1] + (n,)
    out_arr = zeros(out_shape, dtype=a.dtype)
    out_arr[:, :n] = a
    return _ifftn_numpy(out_arr).real * np.prod(out_shape)


def _irfftn_adj_numpy(a):
    n = a.shape[-1] // 2 + 1
    out_arr = _fftn_numpy(a, axis=-1) / a.shape[-1]
    if a.shape[-1] % 2 == 0:  # even
        out_arr[:, 1:n - 1] += np.conj(out_arr[:, :n-1:-1])
    else:  # odd
        out_arr[:, 1:n] += np.conj(out_arr[:, :n-1:-1])
    axes = tuple(range(len(out_arr.shape[:-1])))
    return _fftn_numpy(out_arr[:, :n], axes=axes) / np.prod(out_arr.shape[:-1])


# FFTW functions:
def _fftn_fftw(a, s=None, axes=None):
    if a.dtype not in (FLOAT, COMPLEX):
        raise TypeError('Wrong input type!')
    fftw = PLANS.lookup_fftw('fftn', a, s, axes, NTHREADS)
    if fftw is None:
        fftw = pyfftw.builders.fftn(a, s, axes)
        PLANS.add_fftw('fftn', fftw, s, axes, NTHREADS)
    return fftw(a).copy()


def _ifftn_fftw(a, s=None, axes=None):
    if a.dtype not in (FLOAT, COMPLEX):
        raise TypeError('Wrong input type!')
    fftw = PLANS.lookup_fftw('ifftn', a, s, axes, NTHREADS)
    if fftw is None:
        fftw = pyfftw.builders.ifftn(a, s, axes)
        PLANS.add_fftw('ifftn', fftw, s, axes, NTHREADS)
    return fftw(a).copy()


def _rfftn_fftw(a, s=None, axes=None):
    if a.dtype != FLOAT:
        raise TypeError('Wrong input type!')
    fftw = PLANS.lookup_fftw('rfftn', a, s, axes, NTHREADS)
    if fftw is None:
        fftw = pyfftw.builders.rfftn(a, s, axes)
        PLANS.add_fftw('rfftn', fftw, s, axes, NTHREADS)
    return fftw(a).copy()


def _irfftn_fftw(a, s=None, axes=None):
    if a.dtype != COMPLEX:
        raise TypeError('Wrong input type!')
    fftw = PLANS.lookup_fftw('irfftn', a, s, axes, NTHREADS)
    if fftw is None:
        fftw = pyfftw.builders.irfftn(a, s, axes)
        PLANS.add_fftw('irfftn', fftw, s, axes, NTHREADS)
    return fftw(a).copy()


def _rfftn_adj_fftw(a):
    n = 2 * (a.shape[-1] - 1)
    out_shape = a.shape[:-1] + (n,)
    out_arr = zeros(out_shape, dtype=a.dtype)
    out_arr[:, :a.shape[-1]] = a
    return _ifftn_fftw(out_arr).real * np.prod(out_shape)


def _irfftn_adj_fftw(a):
    out_arr = _fftn_fftw(a, axes=(-1,)) / a.shape[-1]  # FFT of last axis
    n = a.shape[-1] // 2 + 1
    if a.shape[-1] % 2 == 0:  # even
        out_arr[:, 1:n-1] += np.conj(out_arr[:, :n-1:-1])
    else:  # odd
        out_arr[:, 1:n] += np.conj(out_arr[:, :n-1:-1])
    axes = tuple(range(len(out_arr.shape[:-1])))
    return _fftn_fftw(out_arr[:, :n], axes=axes) / np.prod(out_arr.shape[:-1])


# These wisdom functions do nothing if pyFFTW is not available
def dump_wisdom(fname):
    # TODO: Docstring!
    if pyfftw is not None:
        with open(fname, 'wb') as fp:
            pickle.dump(pyfftw.export_wisdom(), fp, pickle.HIGHEST_PROTOCOL)


def load_wisdom(fname):
    # TODO: Docstring!
    if pyfftw is not None:
        if not os.path.exists(fname):
            print("Warning: Wisdom file does not exist. First time use?")
        else:
            with open(fname, 'rb') as fp:
                pyfftw.import_wisdom(pickle.load(fp))


# Array setups:
def zeros(shape, dtype):
    # TODO: Docstring!
    result = np.zeros(shape, dtype)
    if pyfftw is not None:
        result = pyfftw.n_byte_align(result, pyfftw.simd_alignment)
    return result


def empty(shape, dtype):
    # TODO: Docstring!
    result = np.empty(shape, dtype)
    if pyfftw is not None:
        result = pyfftw.n_byte_align(result, pyfftw.simd_alignment)
    return result


# Configure backend:
def configure_backend(backend):
    """
    Change FFT backend.

    Supported values are "numpy" and "fftw".
    """
    global fftn
    global ifftn
    global rfftn
    global irfftn
    global rfftn_adj
    global irfftn_adj
    global BACKEND
    if backend == "numpy":
        fftn = _fftn_numpy
        ifftn = _ifftn_numpy
        rfftn = _rfftn_numpy
        irfftn = _irfftn_numpy
        rfftn_adj = _rfftn_adj_numpy
        irfftn_adj = _irfftn_adj_numpy
        BACKEND = 'numpy'
    elif backend == "fftw":
        if pyfftw is not None:
            fftn = _fftn_fftw
            ifftn = _ifftn_fftw
            rfftn = _rfftn_fftw
            irfftn = _irfftn_fftw
            rfftn_adj = _rfftn_adj_fftw
            irfftn_adj = _irfftn_adj_fftw
            BACKEND = 'pyfftw'
        else:
            print("Error: FFTW requested but not available")


# On import:
if pyfftw is not None:
    configure_backend("fftw")
else:
    configure_backend("numpy")

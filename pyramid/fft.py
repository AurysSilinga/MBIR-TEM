# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Custom FFT module with numpy and FFTW support.

This module provides custom methods for FFTs including inverse, adjoint and real variants. The
FFTW library is supported and is used as a default if the import succeeds. Otherwise the numpy.fft
pack will be used. FFTW objects are saved in a cache after creation which speeds up further similar
FFT operations.

"""


import pickle
import logging
import os

import numpy as np

_log = logging.getLogger(__name__)

try:
    import pyfftw
    BACKEND = 'fftw'
except ImportError:
    pyfftw = None
    BACKEND = 'numpy'
    _log.info('pyFFTW module not found. Using numpy implementation.')

try:
    import multiprocessing
    NTHREADS = multiprocessing.cpu_count()
    del multiprocessing
except ImportError:
    NTHREADS = 1
    _log.info('multiprocessing module not found. Using single core.')


__all__ = ['plans', 'FLOAT', 'COMPLEX', 'dump_wisdom', 'load_wisdom',
           'zeros', 'empty', 'ones', 'configure_backend',
           'fftn', 'ifftn', 'rfftn', 'irfftn', 'rfftn_adj', 'irfftn_adj']


class FFTWCache(object):
    """Class for adding FFTW Plans and on-demand lookups.

    This class is instantiated in this module to store FFTW plans and for the lookup of the former.

    Attributes
    ----------
    cache: dict
        Cache for storing the FFTW plans.

    Notes
    -----
    This class is used internally and is not normally not intended to be used directly by the user.

    """

    _log = logging.getLogger(__name__ + '.FFTWCache')

    def __init__(self):
        self._log.debug('Calling __init__')
        self.cache = dict()
        self._log.debug('Created ' + str(self))

    def add_fftw(self, fft_type, fftw_obj, s, axes, nthreads):
        """Add an FFTW object to the cache.

        Parameters
        ----------
        fft_type: basestring
            Identifier sting for the FFT type ('fftn', 'ifftn', 'rfftn', 'irfftn').
        fftw_obj: :class:`~pyfftw.FFTW` object
            The FFTW object which should be added to the cache.
        s: tuple of ints
            Shape of the output array.
        axes: tuple of ints
            The axes along which the FFTW should be executed.
        nthreads: int
            Number of threads which should be used.

        """
        self._log.debug('Calling add_fftw')
        in_arr = fftw_obj.get_input_array()
        key = (fft_type, in_arr.shape, in_arr.dtype, s, axes, nthreads)
        self.cache[key] = fftw_obj

    def lookup_fftw(self, fft_type, in_arr, s, axes, nthreads):
        """

        Parameters
        ----------
        fft_type: basestring
            Identifier sting for the FFT type ('fftn', 'ifftn', 'rfftn', 'irfftn').
        in_arr:
            Input array, internally, just the `dtype` and the `shape` are used to identify the FFT.
        s: tuple of ints
            Shape of the output array.
        axes: tuple of ints
            The axes along which the FFTW should be executed.
        nthreads: int
            Number of threads which should be used.

        Returns
        -------
        fftw_obj: :class:`~pyfftw.FFTW` object
            The requested FFTW object.

        """
        self._log.debug('Calling lookup_fftw')
        key = (fft_type, in_arr.shape, in_arr.dtype, s, axes, nthreads)
        return self.cache.get(key, None)

    def clear_cache(self):
        """Clear the cache."""
        self._log.debug('Calling clear_cache')
        self.cache = dict()


plans = FFTWCache()
FLOAT = np.float32  # One convenient place to
COMPLEX = np.complex64  # change from 32 to 64 bit


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
    out_arr[:, :a.shape[1]] = a
    return _ifftn_numpy(out_arr).real * np.prod(out_shape)


def _irfftn_adj_numpy(a):
    n = a.shape[-1] // 2 + 1
    out_arr = _fftn_numpy(a, axes=(-1,)) / a.shape[-1]
    if a.shape[-1] % 2 == 0:  # even
        out_arr[:, 1:n - 1] += np.conj(out_arr[:, :n - 1:-1])
    else:  # odd
        out_arr[:, 1:n] += np.conj(out_arr[:, :n - 1:-1])
    axes = tuple(range(len(out_arr.shape[:-1])))
    return _fftn_numpy(out_arr[:, :n], axes=axes) / np.prod(out_arr.shape[:-1])


# FFTW functions:

def _fftn_fftw(a, s=None, axes=None):
    fftw = plans.lookup_fftw('fftn', a, s, axes, NTHREADS)
    if fftw is None:
        fftw = pyfftw.builders.fftn(a, s, axes, threads=NTHREADS)
        plans.add_fftw('fftn', fftw, s, axes, NTHREADS)
    return fftw(a).copy()


def _ifftn_fftw(a, s=None, axes=None):
    fftw = plans.lookup_fftw('ifftn', a, s, axes, NTHREADS)
    if fftw is None:
        fftw = pyfftw.builders.ifftn(a, s, axes, threads=NTHREADS)
        plans.add_fftw('ifftn', fftw, s, axes, NTHREADS)
    return fftw(a).copy()


def _rfftn_fftw(a, s=None, axes=None):
    fftw = plans.lookup_fftw('rfftn', a, s, axes, NTHREADS)
    if fftw is None:
        fftw = pyfftw.builders.rfftn(a, s, axes, threads=NTHREADS)
        plans.add_fftw('rfftn', fftw, s, axes, NTHREADS)
    return fftw(a).copy()


def _irfftn_fftw(a, s=None, axes=None):
    fftw = plans.lookup_fftw('irfftn', a, s, axes, NTHREADS)
    if fftw is None:
        fftw = pyfftw.builders.irfftn(a, s, axes, threads=NTHREADS)
        plans.add_fftw('irfftn', fftw, s, axes, NTHREADS)
    return fftw(a).copy()


def _rfftn_adj_fftw(a):
    # Careful: just works for even a (which is guaranteed by the kernel!)
    n = 2 * (a.shape[-1] - 1)
    out_shape = a.shape[:-1] + (n,)
    out_arr = zeros(out_shape, dtype=a.dtype)
    out_arr[:, :a.shape[-1]] = a
    return _ifftn_fftw(out_arr).real * np.prod(out_shape)


def _irfftn_adj_fftw(a):
    out_arr = _fftn_fftw(a, axes=(-1,)) / a.shape[-1]  # FFT of last axis
    n = a.shape[-1] // 2 + 1
    if a.shape[-1] % 2 == 0:  # even
        out_arr[:, 1:n - 1] += np.conj(out_arr[:, :n - 1:-1])
    else:  # odd
        out_arr[:, 1:n] += np.conj(out_arr[:, :n - 1:-1])
    axes = tuple(range(len(out_arr.shape[:-1])))
    return _fftn_fftw(out_arr[:, :n], axes=axes) / np.prod(out_arr.shape[:-1])


# These wisdom functions do nothing if pyFFTW is not available:

def dump_wisdom(fname):
    """Wrapper function for the pyfftw.export_wisdom(), which uses a pickle dump.

    Parameters
    ----------
    fname: string
        Name of the file in which the wisdom is saved.

    Returns
    -------
    None

    """
    _log.debug('Calling dump_wisdom')
    if pyfftw is not None:
        with open(fname, 'wb') as fp:
            pickle.dump(pyfftw.export_wisdom(), fp, pickle.HIGHEST_PROTOCOL)


def load_wisdom(fname):
    """Wrapper function for the pyfftw.import_wisdom(), which uses a pickle to load a file.

    Parameters
    ----------
    fname: string
        Name of the file from which the wisdom is loaded.

    Returns
    -------
    None

    """
    _log.debug('Calling load_wisdom')
    if pyfftw is not None:
        if not os.path.exists(fname):
            print("Warning: Wisdom file does not exist. First time use?")
        else:
            with open(fname, 'rb') as fp:
                pyfftw.import_wisdom(pickle.load(fp))


# Array setups:
def empty(shape, dtype=FLOAT):
    """Return a new array of given shape and type without initializing entries.

    Parameters
    ----------
    shape: int or tuple of int
        Shape of the array.
    dtype: data-type, optional
        Desired output data-type.

    Returns
    -------
    out: :class:`~numpy.ndarray`
        The created array.

    """
    _log.debug('Calling empty')
    result = np.empty(shape, dtype)
    if pyfftw is not None:
        result = pyfftw.n_byte_align(result, pyfftw.simd_alignment)
    return result


def zeros(shape, dtype=FLOAT):
    """Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape: int or tuple of int
        Shape of the array.
    dtype: data-type, optional
        Desired output data-type.

    Returns
    -------
    out: :class:`~numpy.ndarray`
        The created array.

    """
    _log.debug('Calling zeros')
    result = np.zeros(shape, dtype)
    if pyfftw is not None:
        result = pyfftw.n_byte_align(result, pyfftw.simd_alignment)
    return result


def ones(shape, dtype=FLOAT):
    """Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape: int or tuple of int
        Shape of the array.
    dtype: data-type, optional
        Desired output data-type.

    Returns
    -------
    out: :class:`~numpy.ndarray`
        The created array.

    """
    _log.debug('Calling ones')
    result = np.ones(shape, dtype)
    if pyfftw is not None:
        result = pyfftw.n_byte_align(result, pyfftw.simd_alignment)
    return result


# Configure backend:
def configure_backend(backend):
    """Change FFT backend.

    Parameters
    ----------
    backend: string
        Backend to use. Supported values are "numpy" and "fftw".

    Returns
    -------
    None

    """
    _log.debug('Calling configure_backend')
    global fftn
    global ifftn
    global rfftn
    global irfftn
    global rfftn_adj
    global irfftn_adj
    global BACKEND
    if backend == 'numpy':
        fftn = _fftn_numpy
        ifftn = _ifftn_numpy
        rfftn = _rfftn_numpy
        irfftn = _irfftn_numpy
        rfftn_adj = _rfftn_adj_numpy
        irfftn_adj = _irfftn_adj_numpy
        BACKEND = 'numpy'
    elif backend == 'fftw':
        if pyfftw is not None:
            fftn = _fftn_fftw
            ifftn = _ifftn_fftw
            rfftn = _rfftn_fftw
            irfftn = _irfftn_fftw
            rfftn_adj = _rfftn_adj_fftw
            irfftn_adj = _irfftn_adj_fftw
            BACKEND = 'pyfftw'
        else:
            print('Error: FFTW requested but not available')


# On import:
ifftn = None
fftn = None
rfftn = None
irfftn = None
rfftn_adj = None
irfftn_adj = None
if pyfftw is not None:
    configure_backend('fftw')
else:
    configure_backend('numpy')

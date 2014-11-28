# -*- coding: utf-8 -*-
"""This module provides the :class:`~.Kernel` class, representing the phase contribution of one
single magnetized pixel."""


import numpy as np

import logging


__all__ = ['Kernel']

PHI_0 = 2067.83  # magnetic flux in T*nm²


class Kernel(object):

    '''Class for calculating kernel matrices for the phase calculation.

    Represents the phase of a single magnetized pixel for two orthogonal directions (`u` and `v`),
    which can be accessed via the corresponding attributes. The default elementary geometry is
    `disc`, but can also be specified as the phase of a `slab` representation of a single
    magnetized pixel. During the construction, a few attributes are calculated that are used in
    the convolution during phase calculation in the different :class:`~Phasemapper` classes.
    An instance of the :class:`~.Kernel` class can be called as a function with a `vector`,
    which represents the projected magnetization onto a 2-dimensional grid.

    Attributes
    ----------
    a : float
        The grid spacing in nm.
    dim_uv : tuple of int (N=2), optional
        Dimensions of the 2-dimensional projected magnetization grid from which the phase should
        be calculated.
    b_0 : float, optional
        Saturation magnetization in Tesla, which is used for the phase calculation. Default is 1.
    geometry : {'disc', 'slab'}, optional
        The elementary geometry of the single magnetized pixel.
    u : :class:`~numpy.ndarray` (N=3)
        The phase contribution of one pixel magnetized in u-direction.
    v : :class:`~numpy.ndarray` (N=3)
        The phase contribution of one pixel magnetized in v-direction.
    u_fft : :class:`~numpy.ndarray` (N=3)
        The real FFT of the phase contribution of one pixel magnetized in u-direction.
    v_fft : :class:`~numpy.ndarray` (N=3)
        The real FFT of the phase contribution of one pixel magnetized in v-direction.
    dim_fft : tuple of int (N=2)
        Dimensions of the grid, which is used for the FFT. Calculated by adding the dimensions
        `dim_uv` of the magnetization grid and the dimensions of the kernel (given by
        ``2*dim_uv-1``)
        and increasing to the next multiple of 2 (for faster FFT).
    slice_fft : tuple (N=2) of :class:`slice`
        A tuple of :class:`slice` objects to extract the original field of view from the increased
        size (`size_fft`) of the grid for the FFT-convolution.

    '''  # TODO: overview what all dim_??? mean! and use_fftw, slice(_fft), etc.

    _log = logging.getLogger(__name__+'.Kernel')

    def __init__(self, a, dim_uv, b_0=1., geometry='disc', use_fftw=True, threads=1):
        self._log.debug('Calling __init__')
        # Set basic properties:
        self.dim_uv = dim_uv  # Dimensions of the FOV
        self.dim_kern = tuple(2*np.array(dim_uv)-1)  # Dimensions of the kernel
#        self.size = np.prod(dim_uv)  # TODO: is this even used? (Pixel count)
        self.a = a
        self.geometry = geometry
        # Set up FFT:
        if not use_fftw:
            self.dim_pad = tuple(2**np.ceil(np.log2(2*np.array(dim_uv))).astype(int))  # pow(2)
            self.use_fftw = False
        else:
            try:
                import pyfftw
                self.dim_pad = tuple(2*np.array(dim_uv))  # is at least even (not nec. power of 2)
                self.use_fftw = True
            except ImportError:
                self.dim_pad = tuple(2**np.ceil(np.log2(2*np.array(dim_uv))).astype(int))  # pow(2)
                self.use_fftw = False
                self._log.info('pyFFTW could not be imported, using numpy instead!')
        self.dim_fft = (self.dim_pad[0], self.dim_pad[1]/2+1)  # last axis is real
        self.slice_phase = (slice(dim_uv[0]-1, self.dim_kern[0]),  # Shift because kernel center
                            slice(dim_uv[1]-1, self.dim_kern[1]))  # is not at (0, 0)!
        self.slice_mag = (slice(0, dim_uv[0]),  # Magnetization is padded on the far end!
                          slice(0, dim_uv[1]))  # (Phase cutout is shifted as listed above)
        self.u = np.zeros(self.dim_kern, dtype=np.float32)
        self.v = np.zeros(self.dim_kern, dtype=np.float32)
        self.u_fft = np.zeros(self.dim_fft, dtype=np.complex64)
        self.v_fft = np.zeros(self.dim_fft, dtype=np.complex64)
        if self.use_fftw:  # use pyfftw (FFTW wrapper for python)
            self.u = pyfftw.n_byte_align(self.u, pyfftw.simd_alignment)
            self.v = pyfftw.n_byte_align(self.v, pyfftw.simd_alignment)
            self.u_fft = pyfftw.n_byte_align(self.u_fft, pyfftw.simd_alignment)
            self.v_fft = pyfftw.n_byte_align(self.v_fft, pyfftw.simd_alignment)
            rfftn = pyfftw.builders.rfftn(self.u, self.dim_pad, threads=threads)
            self.threads = threads
        else:  # otherwise use numpy
            rfftn = lambda x: np.fft.rfftn(x, self.dim_pad)
        # Calculate kernel (single pixel phase):
        coeff = b_0 * a**2 / (2*PHI_0)   # Minus is gone because of negative z-direction
        v_dim, u_dim = dim_uv
        u = np.linspace(-(u_dim-1), u_dim-1, num=2*u_dim-1)
        v = np.linspace(-(v_dim-1), v_dim-1, num=2*v_dim-1)
        uu, vv = np.meshgrid(u, v)
        self.u[...] = coeff * self._get_elementary_phase(geometry, uu, vv, a)
        self.v[...] = coeff * self._get_elementary_phase(geometry, vv, uu, a)
        # Calculate Fourier trafo of kernel components:
        self.u_fft[...] = rfftn(self.u)
        self.v_fft[...] = rfftn(self.v)
        self._log.debug('Created '+str(self))

        # TODO: make pyfftw optional (SLOW if kernel has to be build every time like in pm()!)
        # TODO: test if prior build of kernel brings speed up in test_method() or test_fftw()
        # TODO: implement fftw also in phasemapper (JUST there, here: FFT TWICE and big overhead)
        # TODO: BUT allocation of u/v/u_fft/v_fft could be beneficial (try useing with numpy.fft)
        # TODO: Set plan manually? Save computation time also for kernel?
        # TODO: Multithreading?
        # TODO: TakeTime multiple runs?

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(a=%r, dim_uv=%r, geometry=%r)' % \
            (self.__class__, self.a, self.dim_uv, self.geometry)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'Kernel(a=%s, dim_uv=%s, geometry=%s)' % \
            (self.a, self.dim_uv, self.geometry)

    def _get_elementary_phase(self, geometry, n, m, a):
        # TODO: Docstring! Function for the phase of an elementary geometry:
        if geometry == 'disc':
            in_or_out = np.logical_not(np.logical_and(n == 0, m == 0))
            return m / (n**2 + m**2 + 1E-30) * in_or_out
        elif geometry == 'slab':
            def F_a(n, m):
                A = np.log(a**2 * (n**2 + m**2))
                B = np.arctan(n / m)
                return n*A - 2*n + 2*m*B
            return 0.5 * (F_a(n-0.5, m-0.5) - F_a(n+0.5, m-0.5)
                          - F_a(n-0.5, m+0.5) + F_a(n+0.5, m+0.5))

    def print_info(self):
        print 'Shape of the FOV   :', self.dim_uv
        print 'Shape of the Kernel:', self.dim_kern
        print 'Zero-padded shape  :', self.dim_pad
        print 'Shape of the FFT   :', self.dim_fft
        print 'Slice for the phase:', self.slice_phase
        print 'Slice for the magn.:', self.slice_mag
        print 'Grid spacing: {} nm'.format(self.a)
        print 'Geometry:', self.geometry
        print 'Use FFTW: {}; with {} thread(s)'.format(self.use_fftw, self.threads)

# -*- coding: utf-8 -*-
"""This module provides the :class:`~.Kernel` class, representing the phase contribution of one
single magnetized pixel."""


import numpy as np

import logging


PHI_0 = -2067.83    # magnetic flux in T*nm²


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
    numcore : boolean, optional
        Boolean choosing if Cython enhanced routines from the :mod:`~.pyramid.numcore` module
        should be used. Default is True.
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

    '''

    LOG = logging.getLogger(__name__+'.Kernel')

    def __init__(self, a, dim_uv, b_0=1., numcore=True, geometry='disc'):
        self.LOG.debug('Calling __init__')

        # Function for the phase of an elementary geometry:
        def get_elementary_phase(geometry, n, m, a):
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

        # Set basic properties:
        self.dim_uv = dim_uv  # Size of the FOV, not the kernel (kernel is bigger)!
        self.size = np.prod(dim_uv)  # Pixel count
        self.a = a
        self.numcore = numcore
        self.geometry = geometry
        # Calculate kernel (single pixel phase):
        coeff = -b_0 * a**2 / (2*PHI_0)
        v_dim, u_dim = dim_uv
        u = np.linspace(-(u_dim-1), u_dim-1, num=2*u_dim-1)
        v = np.linspace(-(v_dim-1), v_dim-1, num=2*v_dim-1)
        uu, vv = np.meshgrid(u, v)
        self.u = coeff * get_elementary_phase(geometry, uu, vv, a)
        self.v = coeff * get_elementary_phase(geometry, vv, uu, a)
        # Calculate Fourier trafo of kernel components:
        dim_combined = 3*np.array(dim_uv) - 2  # (dim_uv-1) + dim_uv + (dim_uv-1) mag + kernel
        self.dim_fft = 2 ** np.ceil(np.log2(dim_combined)).astype(int)  # next multiple of 2
        self.slice_fft = (slice(dim_uv[0]-1, 2*dim_uv[0]-1), slice(dim_uv[1]-1, 2*dim_uv[1]-1))
        self.slice_fft_compl = (slice(2*dim_uv[0]-1, 2*dim_uv[0]-1), slice(2*dim_uv[1]-1, 2*dim_uv[1]-1))
        self.u_fft = np.fft.rfftn(self.u, self.dim_fft)
        self.v_fft = np.fft.rfftn(self.v, self.dim_fft)
        self.u_fft_compl = np.fft.fftn(self.u, self.dim_fft)
        self.v_fft_compl = np.fft.fftn(self.v, self.dim_fft)
        self.LOG.debug('Created '+str(self))

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(a=%r, dim_uv=%r, numcore=%r, geometry=%r)' % \
            (self.__class__, self.a, self.dim_uv, self.numcore, self.geometry)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'Kernel(a=%s, dim_uv=%s, numcore=%s, geometry=%s)' % \
            (self.a, self.dim_uv, self.numcore, self.geometry)

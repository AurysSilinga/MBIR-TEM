# -*- coding: utf-8 -*-
"""This module provides the :class:`~.Kernel` class, representing the phase contribution of one
single magnetized pixel."""


import logging

import numpy as np

import pyramid.numcore.kernel_core as core


PHI_0 = -2067.83    # magnetic flux in T*nm²
# TODO: sign?


class Kernel(object):

    '''Class for calculating kernel matrices for the phase calculation.
    
    
    
    This module provides the :class:`~.Kernel` class whose instances can be used to calculate and
    store the kernel matrix representing the phase of a single pixel for the convolution used in the
    phase calculation. The phasemap of a single pixel for two orthogonal directions (`u` and `v`) are
    stored seperately as 2-dimensional matrices. The Jacobi matrix of the phasemapping just depends
    on the kernel and can be calculated via the :func:`~.get_jacobi` function. Storing the Jacobi
    matrix uses much memory, thus it is also possible to directly get the multiplication of a given
    vector with the (transposed) Jacobi matrix without explicit calculation of the latter.
    It is possible to load data from and save them to NetCDF4 files. See :class:`~.Kernel` for further
    information.
    

    Represents the phase of a single magnetized pixel for two orthogonal directions (`u` and `v`),
    which can be accessed via the corresponding attributes. The default elementary geometry is
    `disc`, but can also be specified as the phase of a `slab` representation of a single
    magnetized pixel. During the construction, a few attributes are calculated that are used in
    the convolution during phase calculation.


    An instance `kernel` of the :class:`~.Kernel` class is callable via:
    
    .. :function:: kernel(vector)
        
        do stuff
        
        :param str sender: do other stuff
        :return: nix

    with `vector` being a :class:`~numpy.ndarray` (N=1).

    Attributes
    ----------
    dim_uv : tuple (N=2)
        Dimensions of the projected magnetization grid.
    a : float
        The grid spacing in nm.
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
    dim_fft : tuple (N=2)
        Dimensions of the grid, which is used for the FFT. Calculated by adding the dimensions
        `dim_uv` of the magnetization grid and the dimensions of the kernel (given by
        ``2*dim_uv-1``)
        and increasing to the next multiple of 2 (for faster FFT).
    slice_fft : tuple (N=2) of :class:`slice`
        A tuple of :class:`slice` objects to extract the original field of view from the increased
        size (`size_fft`) of the grid for the FFT-convolution.

    '''# TODO: Can be used for several PhaseMappers via the fft arguments or via calling!
    
    def __init__(self, a, dim_uv, numcore=True, geometry='disc'):
        '''Constructor for a :class:`~.Kernel` object for representing a kernel matrix.

        Parameters
        ----------
        a : float
            The grid spacing in nm.
        dim_uv : tuple (N=2)
            Dimensions of the projected magnetization grid.
        geometry : {'disc', 'slab'}, optional
            The elementary geometry of the single magnetized pixel.

        ''' # TODO: Docstring
        self.log = logging.getLogger(__name__)
        self.log.info('Calling __init__')
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
        self.dim_uv = dim_uv  # !!! size of the FOV, not the kernel (kernel is bigger)!
        self.a = a
        self.numcore = numcore
        self.geometry = geometry
        # Calculate kernel (single pixel phase):
        coeff = -a**2 / (2*PHI_0)
        v_dim, u_dim = dim_uv
        u = np.linspace(-(u_dim-1), u_dim-1, num=2*u_dim-1)
        v = np.linspace(-(v_dim-1), v_dim-1, num=2*v_dim-1)
        uu, vv = np.meshgrid(u, v)
        self.u = coeff * get_elementary_phase(geometry, uu, vv, a)
        self.v = coeff * get_elementary_phase(geometry, vv, uu, a)
        # Calculate Fourier trafo of kernel components:
        dim_combined = 3*np.array(dim_uv) - 1  # dim_uv + (2*dim_uv - 1) magnetisation + kernel
        self.dim_fft = 2 ** np.ceil(np.log2(dim_combined)).astype(int)  # next multiple of 2
        self.slice_fft = (slice(dim_uv[0]-1, 2*dim_uv[0]-1), slice(dim_uv[1]-1, 2*dim_uv[1]-1))
        self.u_fft = np.fft.rfftn(self.u, self.dim_fft)
        self.v_fft = np.fft.rfftn(self.v, self.dim_fft)
        self.log.info('Created '+str(self))

    def __call__(self, x):
        '''Test'''
        self.log.info('Calling __call__')
#        print 'Kernel - __call__:', len(x)
        if self.numcore:
            return self._multiply_jacobi_core(x)
        else:
            return self._multiply_jacobi(x)
        # TODO: Bei __init__ variable auf die entsprechende Funktion setzen.

    def __repr__(self):
        self.log.info('Calling __repr__')
        return '%s(a=%r, dim_uv=%r, numcore=%r, geometry=%r)' % \
            (self.__class__, self.a, self.dim_uv, self.numcore, self.geometry)

    def jac_dot(self, vector):
        '''TEST'''# TODO: Docstring
        self.log.info('Calling jac_dot')
#        print 'Kernel - jac_dot:', len(vector)
        if self.numcore:
            return self._multiply_jacobi_core(vector)
        else:
            return self._multiply_jacobi(vector)

    def jac_T_dot(self, vector):
        # TODO: Docstring
        self.log.info('Calling jac_dot_T')
#        print 'Kernel - jac_T_dot:', len(vector)
        return self._multiply_jacobi_T(vector)

    def _multiply_jacobi(self, vector):
        '''Calculate the product of the Jacobi matrix with a given `vector`.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized form of the magnetization in `u`- and `v`-direction of every pixel
            (row-wise). The first ``N**2`` elements have to correspond to the `u`-, the next
            ``N**2`` elements to the `v`-component of the magnetization.
        
        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the Jacobi matrix (which is not explicitely calculated) with the vector.

        '''# TODO: move!
        self.log.info('Calling _multiply_jacobi')
        v_dim, u_dim = self.dim_uv
        size = np.prod(self.dim_uv)
        assert len(vector) == 2*size, 'vector size not compatible!'
        result = np.zeros(size)
        for s in range(size):  # column-wise (two columns at a time, u- and v-component)
            i = s % u_dim
            j = int(s/u_dim)
            u_min = (u_dim-1) - i
            u_max = (2*u_dim-1) - i  # = u_min + u_dim
            v_min = (v_dim-1) - j
            v_max = (2*v_dim-1) - j  # = v_min + v_dim
            result += vector[s] * self.u[v_min:v_max, u_min:u_max].reshape(-1)  # u
            result -= vector[s+size] * self.v[v_min:v_max, u_min:u_max].reshape(-1)  # v
        return result

    def _multiply_jacobi_T(self, vector):
        '''Calculate the product of the transposed Jacobi matrix with a given `vector`.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized form of the magnetization in `u`- and `v`-direction of every pixel
            (row-wise). The first ``N**2`` elements have to correspond to the `u`-, the next
            ``N**2`` elements to the `v`-component of the magnetization.
        
        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the transposed Jacobi matrix (which is not explicitely calculated) with
            the vector.

        '''# TODO: move!
        self.log.info('Calling _multiply_jacobi_T')
        v_dim, u_dim = self.dim_uv
        size = np.prod(self.dim_uv)
        assert len(vector) == size, 'vector size not compatible! vector: {}, size: {}'.format(len(vector),size)
        result = np.zeros(2*size)
        for s in range(size):  # row-wise (two rows at a time, u- and v-component)
            i = s % u_dim
            j = int(s/u_dim)
            u_min = (u_dim-1) - i
            u_max = (2*u_dim-1) - i
            v_min = (v_dim-1) - j
            v_max = (2*v_dim-1) - j
            result[s] = np.sum(vector*self.u[v_min:v_max, u_min:u_max].reshape(-1))
            result[s+size] = np.sum(vector*-self.v[v_min:v_max, u_min:u_max].reshape(-1))
        return result

    def _multiply_jacobi_core(self, vector):
        self.log.info('Calling _multiply_jacobi_core')
        result = np.zeros(np.prod(self.dim_uv))
        core.multiply_jacobi_core(self.dim_uv[0], self.dim_uv[1], self.u, self.v, vector, result)
        return result

# -*- coding: utf-8 -*-
"""Class for the calculation and storage of kernel.

This module provides the :class:`~.Kernel` class whose instances can be used to calculate and
store the kernel matrix representing the phase of a single pixel for the convolution used in the
phase calculation. The phasemap of a single pixel for two orthogonal directions (`u` and `v`) are
stored seperately as 2-dimensional matrices. The Jacobi matrix of the phasemapping just depends
on the kernel and can be calculated via the :func:`~.get_jacobi` function. Storing the Jacobi
matrix uses much memory, thus it is also possible to directly get the multiplication of a given
vector with the (transposed) Jacobi matrix without explicit calculation of the latter.
It is possible to load data from and save them to NetCDF4 files. See :class:`~.Kernel` for further
information.

"""


import numpy as np


PHI_0 = -2067.83    # magnetic flux in T*nm²


class Kernel:
    '''Class for calculating kernel matrices for the phase calculation.

    Represents the phase of a single magnetized pixel for two orthogonal directions (`u` and `v`),
    which can be accessed via the corresponding attributes. The default elementary geometry is
    `disc`, but can also be specified as the phase of a `slab` representation of a single
    magnetized pixel. During the construction, a few attributes are calculated that are used in
    the convolution during phase calculation.

    Attributes
    ----------
    dim : tuple (N=2)
        Dimensions of the projected magnetization grid.
    a : float
        The grid spacing in nm.
    geometry : {'disc', 'slab'}, optional
        The elementary geometry of the single magnetized pixel.
    b_0 : float, optional
        The saturation magnetic induction. Default is 1.
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
        `dim` of the magnetization grid and the dimensions of the kernel (given by ``2*dim-1``)
        and increasing to the next multiple of 2 (for faster FFT).
    slice_fft : tuple (N=2) of :class:`slice`
        A tuple of :class:`slice` objects to extract the original field of view from the increased
        size (size_fft) of the grid for the FFT-convolution.

    '''
    
    def __init__(self, dim, a, b_0=1, geometry='disc'):
        '''Constructor for a :class:`~.Kernel` object for representing a kernel matrix.

        Parameters
        ----------
        dim : tuple (N=2)
            Dimensions of the projected magnetization grid.
        a : float
            The grid spacing in nm.
        b_0 : float, optional
            The saturation magnetic induction. Default is 1.
        geometry : {'disc', 'slab'}, optional
            The elementary geometry of the single magnetized pixel.

        '''
        # Function for the phase of an elementary geometry:
        def get_elementary_phase(geometry, n, m, res):
            if geometry == 'disc':
                in_or_out = np.logical_not(np.logical_and(n == 0, m == 0))
                return m / (n**2 + m**2 + 1E-30) * in_or_out
            elif geometry == 'slab':
                def F_h(n, m):
                    a = np.log(res**2 * (n**2 + m**2))
                    b = np.arctan(n / m)
                    return n*a - 2*n + 2*m*b
                return 0.5 * (F_h(n-0.5, m-0.5) - F_h(n+0.5, m-0.5)
                            - F_h(n-0.5, m+0.5) + F_h(n+0.5, m+0.5))
        # Set basic properties:
        self.dim = dim  # !!! size of the FOV, not the kernel (kernel is bigger)!
        self.a = a
        self.geometry = geometry
        self.b_0 = b_0
        # Calculate kernel (single pixel phase):
        coeff = -a**2 / (2*PHI_0)
        v_dim, u_dim = dim
        u = np.linspace(-(u_dim-1), u_dim-1, num=2*u_dim-1)
        v = np.linspace(-(v_dim-1), v_dim-1, num=2*v_dim-1)
        uu, vv = np.meshgrid(u, v)
        self.u = coeff * get_elementary_phase(geometry, uu, vv, a)
        self.v = coeff * get_elementary_phase(geometry, vv, uu, a)
        # Calculate Fourier trafo of kernel components:
        dim_combined = 3*np.array(dim) - 1  # dim + (2*dim - 1) magnetisation + kernel
        self.dim_fft = 2 ** np.ceil(np.log2(dim_combined)).astype(int)  # next multiple of 2
        self.slice_fft = (slice(dim[0]-1, 2*dim[0]-1), slice(dim[1]-1, 2*dim[1]-1))
        self.u_fft = np.fft.rfftn(self.u, self.dim_fft)
        self.v_fft = np.fft.rfftn(self.v, self.dim_fft)

    def get_jacobi(self):
        '''Calculate the Jacobi matrix for the phase calculation from a projected magnetization.

        Parameters
        ----------
        None
        
        Returns
        -------
        jacobi : :class:`~numpy.ndarray` (N=2)
            Jacobi matrix containing the derivatives of the phase at every pixel with respect to
            the projected magetization. Has `N` columns for the `u`-component of the magnetization
            and `N` columns for the `v`-component (from left to right) and ``N**2`` rows for the
            phase at every pixel.

        Notes
        -----
        Just use for small dimensions, Jacobi Matrix scales with order of ``N**4``.

        '''
        v_dim, u_dim = self.dim
        jacobi = np.zeros((v_dim*u_dim, 2*v_dim*u_dim))  
#       nc.get_jacobi_core(dim[0], dim[1], v_phi, u_phi, jacobi)
#       return jacobi
        for j in range(v_dim):
            for i in range(u_dim):
                u_column = i + u_dim*j
                v_column = i + u_dim*j + u_dim*v_dim
                u_min = (u_dim-1) - i
                u_max = (2*u_dim-1) - i
                v_min = (v_dim-1) - j
                v_max = (2*v_dim-1) - j
                # u_dim*v_dim columns for the u-component:
                jacobi[:, u_column] = self.u[v_min:v_max, u_min:u_max].reshape(-1)
                # u_dim*v_dim columns for the v-component (note the minus!):
                jacobi[:, v_column] = -self.v[v_min:v_max, u_min:u_max].reshape(-1)
        return jacobi

    def multiply_jacobi(self, vector):
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

        '''
        v_dim, u_dim = self.dim
        size = v_dim * u_dim
        assert len(vector) == 2*size, 'vector size not compatible!'
        result = np.zeros(size)
        for s in range(size):  # column-wise (two columns at a time, u- and v-component)
            i = s % u_dim
            j = int(s/u_dim)
            u_min = (u_dim-1) - i
            u_max = (2*u_dim-1) - i
            v_min = (v_dim-1) - j
            v_max = (2*v_dim-1) - j
            result += vector[s]*self.u[v_min:v_max, u_min:u_max].reshape(-1)  # u
            result += vector[s+size]*-self.v[v_min:v_max, u_min:u_max].reshape(-1)  # v        
        return result

    def multiply_jacobi_T(self, vector):
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

        '''
        v_dim, u_dim = self.dim
        size = v_dim * u_dim
        assert len(vector) == size, 'vector size not compatible!'
        result = np.zeros(2*size)
        for s in range(size):  # row-wise (two rows at a time, u- and v-component)
            i = s % u_dim
            j = int(s/u_dim)
            u_min = (u_dim-1) - i
            u_max = (2*u_dim-1) - i
            v_min = (v_dim-1) - j
            v_max = (2*v_dim-1) - j
            result[s] = np.sum(vector*self.u[v_min:v_max, u_min:u_max].reshape(-1))  # u
            result[s+size] = np.sum(vector*-self.v[v_min:v_max, u_min:u_max].reshape(-1))  # v        
        return result

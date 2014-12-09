# -*- coding: utf-8 -*-
"""This module executes several forward models to calculate the magnetic or electric phase map from
a given projection of a 3-dimensional magnetic distribution (see :mod:`~pyramid.projector`).
For the magnetic phase map, an approach using real space and one using Fourier space is provided.
The electrostatic contribution is calculated by using the assumption of a mean inner potential.

"""


import numpy as np
from numpy import pi

import abc

import pyramid.numcore.phasemapper_core as nc
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap
from pyramid.projector import SimpleProjector
from pyramid.kernel import Kernel
from pyramid import fft

import logging


__all__ = ['PhaseMapperRDFC', 'PhaseMapperRDRC', 'PhaseMapperFDFC', 'pm']
_log = logging.getLogger(__name__)

PHI_0 = 2067.83    # magnetic flux in T*nm²
H_BAR = 6.626E-34  # Planck constant in J*s
M_E = 9.109E-31    # electron mass in kg
Q_E = 1.602E-19    # electron charge in C
C = 2.998E8        # speed of light in m/s


class PhaseMapper(object):

    '''Abstract base class for the phase calculation from a 2-dimensional distribution.

    The :class:`~.PhaseMapper-` class represents a strategy for the phasemapping of a
    2-dimensional magnetic distribution with two components onto a scalar phase map.
    :class:`~.Kernel` is an abstract base class and provides a unified interface which should be
    subclassed with custom :func:`__init__` and :func:`__call__` functions. Concrete subclasses
    can be called as a function and take a :class:`~.MagData` object as input and return a
    :class:`~.PhaseMap` object.

    '''

    __metaclass__ = abc.ABCMeta
    _log = logging.getLogger(__name__+'.PhaseMapper')

    @abc.abstractmethod
    def __call__(self, mag_data):
        raise NotImplementedError()

    @abc.abstractmethod
    def jac_dot(self, vector):
        raise NotImplementedError()

    @abc.abstractmethod
    def jac_T_dot(self, vector):
        raise NotImplementedError()


class PhaseMapperRDFC(PhaseMapper):

    '''Class representing a phase mapping strategy using real space discretization and Fourier
    space convolution.

    The :class:`~.PMConvolve` class represents a phase mapping strategy involving discretization in
    real space. It utilizes the convolution in Fourier space, directly takes :class:`~.MagData`
    objects and returns :class:`~.PhaseMap` objects.

    Attributes
    ----------
    a : float
        The grid spacing in nm.
    projector : :class:`~.Projector`
        Projector which should be used for the projection of the 3-dimensional magnetization
        distribution.
    b_0 : float, optional
        The magnetic induction corresponding to a magnetization `M`\ :sub:`0` in T.
        The default is 1.
    geometry : {'disc', 'slab'}, optional
        Elementary geometry which is used for the phase contribution of one pixel.
        Default is 'disc'.
    m: int
        Size of the image space.
    n: int
        Size of the input space.

    '''

    _log = logging.getLogger(__name__+'.PhaseMapperRDFC')

    def __init__(self, kernel):
        self._log.debug('Calling __init__')
        self.kernel = kernel
        self.m = np.prod(kernel.dim_uv)
        self.n = 2 * self.m
        self.u_mag = fft.zeros(kernel.dim_pad, dtype=fft.FLOAT)
        self.v_mag = fft.zeros(kernel.dim_pad, dtype=fft.FLOAT)
        self.mag_adj = fft.zeros(kernel.dim_pad, dtype=fft.FLOAT)
        self._log.debug('Created '+str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(kernel=%r)' % (self.__class__, self.kernel)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'PhaseMapperRDFC(kernel=%s)' % (self.kernel)

    def __call__(self, mag_data):
        self._log.debug('Calling __call__')
        assert isinstance(mag_data, MagData), 'Only MagData objects can be mapped!'
        assert mag_data.a == self.kernel.a, 'Grid spacing has to match!'
        assert mag_data.dim[0] == 1, 'Magnetic distribution must be 2-dimensional!'
        assert mag_data.dim[1:3] == self.kernel.dim_uv, 'Dimensions do not match!'
        # Process input parameters:
        self.u_mag[self.kernel.slice_mag] = mag_data.magnitude[0, 0, ...]  # u-component
        self.v_mag[self.kernel.slice_mag] = mag_data.magnitude[1, 0, ...]  # v-component
        return PhaseMap(mag_data.a, self._convolve())

    def _convolve(self):
        # Fourier transform the projected magnetisation:
        self.u_mag_fft = fft.rfftn(self.u_mag)
        self.v_mag_fft = fft.rfftn(self.v_mag)
        # Convolve the magnetization with the kernel in Fourier space:
        self.phase_fft = self.u_mag_fft*self.kernel.u_fft - self.v_mag_fft*self.kernel.v_fft
        # Return the result:
        return fft.irfftn(self.phase_fft)[self.kernel.slice_phase]

    def jac_dot(self, vector):
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
        assert len(vector) == self.n, \
            'vector size not compatible! vector: {}, size: {}'.format(len(vector), self.n)
        self.u_mag[self.kernel.slice_mag], self.v_mag[self.kernel.slice_mag] = \
            np.reshape(vector, (2,)+self.kernel.dim_uv)
        result = self._convolve().flatten()
        return result

    def jac_T_dot(self, vector):
        '''Calculate the product of the transposed Jacobi matrix with a given `vector`.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vector with ``N**2`` entries which represents a matrix with dimensions like a scalar
            phasemap.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the transposed Jacobi matrix (which is not explicitely calculated) with
            the vector, which has ``2*N**2`` entries like a 2D magnetic projection.

        '''
        assert len(vector) == self.m, \
            'vector size not compatible! vector: {}, size: {}'.format(len(vector), self.m)
        self.mag_adj[self.kernel.slice_mag] = vector.reshape(self.kernel.dim_uv)
        mag_adj_fft = fft.irfftn_adj(self.mag_adj)
        u_phase_adj_fft = mag_adj_fft * self.kernel.u_fft
        v_phase_adj_fft = mag_adj_fft * -self.kernel.v_fft
        u_phase_adj = fft.rfftn_adj(u_phase_adj_fft)[self.kernel.slice_phase]
        v_phase_adj = fft.rfftn_adj(v_phase_adj_fft)[self.kernel.slice_phase]
        return -np.concatenate((u_phase_adj.flatten(), v_phase_adj.flatten()))  # TODO: why minus?


class PhaseMapperRDRC(PhaseMapper):

    '''Class representing a phase mapping strategy using real space discretization.

    The :class:`~.PMReal` class represents a phase mapping strategy involving discretization in
    real space. It directly takes :class:`~.MagData` objects and returns :class:`~.PhaseMap`
    objects.

    Attributes
    ----------
    a : float
        The grid spacing in nm.
    projector : :class:`~.Projector`
        Projector which should be used for the projection of the 3-dimensional magnetization
        distribution.
    b_0 : float, optional
        The magnetic induction corresponding to a magnetization `M`\ :sub:`0` in T.
        The default is 1.
    threshold : float, optional
        Threshold determining for which voxels the phase contribution will be calculated. The
        default is 0, which means that all voxels with non-zero magnetization will contribute.
        Should be above noise level.
    geometry : {'disc', 'slab'}, optional
        Elementary geometry which is used for the phase contribution of one pixel.
        Default is 'disc'.
    numcore : boolean, optional
        Boolean choosing if Cython enhanced routines from the :mod:`~.pyramid.numcore` module
        should be used. Default is True.

    '''

    _log = logging.getLogger(__name__+'.PhaseMapperRDRC')

    def __init__(self, kernel, threshold=0, numcore=True):
        self._log.debug('Calling __init__')
        self.kernel = kernel
        self.threshold = threshold
        self.numcore = numcore
        self.m = np.prod(kernel.dim_uv)
        self.n = 2 * self.m
        self._log.debug('Created '+str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(kernel=%r, threshold=%r, numcore=%r)' % \
            (self.__class__, self.kernel, self.threshold, self.numcore)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'PhaseMapperRDRC(kernel=%s, threshold=%s, numcore=%s)' % \
            (self.kernel, self.threshold, self.numcore)

    def __call__(self, mag_data):
        self._log.debug('Calling __call__')
        dim_uv = self.kernel.dim_uv
        assert isinstance(mag_data, MagData), 'Only MagData objects can be mapped!'
        assert mag_data.a == self.kernel.a, 'Grid spacing has to match!'
        assert mag_data.dim[0] == 1, 'Magnetic distribution must be 2-dimensional!'
        assert mag_data.dim[1:3] == dim_uv, 'Dimensions do not match!'
        # Process input parameters:
        u_mag, v_mag = mag_data.magnitude[0:2, 0, ...]
        # Get kernel (lookup-tables for the phase of one pixel):
        u_phi = self.kernel.u
        v_phi = self.kernel.v
        # Calculation of the phase:
        phase = np.zeros(dim_uv, dtype=np.float32)
        if self.numcore:
            nc.phasemapper_real_convolve(dim_uv[0], dim_uv[1], v_phi, u_phi,
                                         v_mag, u_mag, phase, self.threshold)
        else:
            for j in range(dim_uv[0]):
                for i in range(dim_uv[1]):
                    u_phase = u_phi[dim_uv[0]-1-j:(2*dim_uv[0]-1)-j,
                                    dim_uv[1]-1-i:(2*dim_uv[1]-1)-i]
                    if abs(u_mag[j, i]) > self.threshold:
                        phase += u_mag[j, i] * u_phase
                    v_phase = v_phi[dim_uv[0]-1-j:(2*dim_uv[0]-1)-j,
                                    dim_uv[1]-1-i:(2*dim_uv[1]-1)-i]
                    if abs(v_mag[j, i]) > self.threshold:
                        phase -= v_mag[j, i] * v_phase
        # Return the phase:
        return PhaseMap(mag_data.a, phase)

    def jac_dot(self, vector):
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
        assert len(vector) == self.n, \
            'vector size not compatible! vector: {}, size: {}'.format(len(vector), self.n)
        v_dim, u_dim = self.kernel.dim_uv
        result = np.zeros(self.m)
        if self.numcore:
            nc.jac_dot_real_convolve(v_dim, u_dim, self.kernel.u, self.kernel.v, vector, result)
        else:
            # Iterate over all contributing pixels (numbered consecutively)
            for s in range(self.m):  # column-wise (two columns at a time, u- and v-component)
                i = s % u_dim  # u-coordinate of current contributing pixel
                j = int(s/u_dim)  # v-coordinate of current ccontributing pixel
                u_min = (u_dim-1) - i  # u_dim-1: center of the kernel
                u_max = (2*u_dim-1) - i  # = u_min + u_dim
                v_min = (v_dim-1) - j  # v_dim-1: center of the kernel
                v_max = (2*v_dim-1) - j  # = v_min + v_dim
                result += vector[s] * self.kernel.u[v_min:v_max, u_min:u_max].reshape(-1)
                result -= vector[s+self.m] * self.kernel.v[v_min:v_max, u_min:u_max].reshape(-1)
        return result

    def jac_T_dot(self, vector):
        '''Calculate the product of the transposed Jacobi matrix with a given `vector`.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vector with ``N**2`` entries which represents a matrix with dimensions like a scalar
            phasemap.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the transposed Jacobi matrix (which is not explicitely calculated) with
            the vector, which has ``2*N**2`` entries like a 2D magnetic projection.

        '''
        assert len(vector) == self.m, \
            'vector size not compatible! vector: {}, size: {}'.format(len(vector), self.m)
        v_dim, u_dim = self.dim_uv
        result = np.zeros(self.n)
        if self.numcore:
            nc.jac_T_dot_real_convolve(v_dim, u_dim, self.kernel.u, self.kernel.v, vector, result)
        else:
            # Iterate over all contributing pixels (numbered consecutively):
            for s in range(self.m):  # row-wise (two rows at a time, u- and v-component)
                i = s % u_dim  # u-coordinate of current contributing pixel
                j = int(s/u_dim)  # v-coordinate of current contributing pixel
                u_min = (u_dim-1) - i  # u_dim-1: center of the kernel
                u_max = (2*u_dim-1) - i  # = u_min + u_dim
                v_min = (v_dim-1) - j  # v_dim-1: center of the kernel
                v_max = (2*v_dim-1) - j  # = v_min + v_dim
                result[s] = np.sum(vector*self.u[v_min:v_max, u_min:u_max].reshape(-1))
                result[s+self.m] = np.sum(vector*-self.v[v_min:v_max, u_min:u_max].reshape(-1))
        return result


class PhaseMapperFDFC(PhaseMapper):

    '''Class representing a phase mapping strategy using a discretization in Fourier space.

    The :class:`~.PMFourier` class represents a phase mapping strategy involving discretization in
    Fourier space. It utilizes the Fourier transforms, which are inherently calculated in the
    :class:`~.Kernel` class and directly takes :class:`~.MagData` objects and returns
    :class:`~.PhaseMap` objects.

    Attributes
    ----------
    a : float
        The grid spacing in nm.
    projector : :class:`~.Projector`
        Projector which should be used for the projection of the 3-dimensional magnetization
        distribution.
    b_0 : float, optional
        The magnetic induction corresponding to a magnetization `M`\ :sub:`0` in T.
        The default is 1.
    padding : int, optional
        Factor for the zero padding. The default is 0 (no padding). For a factor of n the number
        of pixels is increase by ``(1+n)**2``. Padded zeros are cropped at the end.

    '''

    _log = logging.getLogger(__name__+'.PhaseMapperFDFC')

    def __init__(self, a, dim_uv, b_0=1, padding=0):
        self._log.debug('Calling __init__')
        self.a = a
        self.dim_uv = dim_uv
        self.b_0 = b_0
        self.padding = padding
        self.m = np.prod(dim_uv)
        self.n = 2 * self.m
        self._log.debug('Created '+str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(a=%r, dim_uv=%r, b_0=%r, padding=%r)' % \
            (self.__class__, self.a, self.dim_uv, self.b_0, self.padding)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'PhaseMapperFDFC(a=%s, dim_uv=%s, b_0=%s, padding=%s)' % \
            (self.a, self.dim_uv, self.b_0, self.padding)

    def __call__(self, mag_data):
        self._log.debug('Calling __call__')
        assert isinstance(mag_data, MagData), 'Only MagData objects can be mapped!'
        assert mag_data.a == self.a, 'Grid spacing has to match!'
        assert mag_data.dim[0] == 1, 'Magnetic distribution must be 2-dimensional!'
        assert mag_data.dim[1:3] == self.dim_uv, 'Dimensions do not match!'
        v_dim, u_dim = self.dim_uv
        u_mag, v_mag = mag_data.magnitude[0:2, 0, ...]
        # Create zero padded matrices:
        u_pad = int(u_dim/2 * self.padding)
        v_pad = int(v_dim/2 * self.padding)
        u_mag_pad = np.pad(u_mag, ((v_pad, v_pad), (u_pad, u_pad)), 'constant')
        v_mag_pad = np.pad(v_mag, ((v_pad, v_pad), (u_pad, u_pad)), 'constant')
        # Fourier transform of the two components:
        u_mag_fft = np.fft.fftshift(np.fft.rfft2(u_mag_pad), axes=0)
        v_mag_fft = np.fft.fftshift(np.fft.rfft2(v_mag_pad), axes=0)
        # Calculate the Fourier transform of the phase:
        f_nyq = 0.5 / self.a  # nyquist frequency
        f_u = np.linspace(0, f_nyq, u_mag_fft.shape[1])
        f_v = np.linspace(-f_nyq, f_nyq, u_mag_fft.shape[0], endpoint=False)
        f_uu, f_vv = np.meshgrid(f_u, f_v)
        coeff = - (1j*self.b_0*self.a) / (2*PHI_0)  # Minus because of negative z-direction
        phase_fft = coeff * (u_mag_fft*f_vv - v_mag_fft*f_uu) / (f_uu**2 + f_vv**2 + 1e-30)
        # Transform to real space and revert padding:
        phase_pad = np.fft.irfft2(np.fft.ifftshift(phase_fft, axes=0))
        phase = phase_pad[v_pad:v_pad+v_dim, u_pad:u_pad+u_dim]
        return PhaseMap(mag_data.a, phase)

    def jac_dot(self, vector):
        self._log.debug('Calling jac_dot')
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
        assert len(vector) == self.n, \
            'vector size not compatible! vector: {}, size: {}'.format(len(vector), self.n)
        mag_proj = MagData(self.a, np.zeros((3, 1)+self.dim_uv))
        magnitude_proj = self.jac_dot(vector).reshape((2, )+self.dim_uv)
        mag_proj.magnitude[1:3, 0, ...] = magnitude_proj
        # TODO: instead call common subroutine operating on u_mag, v_mag with __call__?
        return self(mag_proj).phase_vec

    def jac_T_dot(self, vector):
        raise NotImplementedError()
        # TODO: Implement!


class PhaseMapperElectric(PhaseMapper):

    '''Class representing a phase mapping strategy for the electrostatic contribution.

    The :class:`~.PMElectric` class represents a phase mapping strategy for the electrostatic
    contribution to the electron phase shift which results e.g. from the mean inner potential in
    certain samples and which is sensitive to properties of the electron microscope. It directly
    takes :class:`~.MagData` objects and returns :class:`~.PhaseMap` objects.

    Attributes
    ----------
    a : float
        The grid spacing in nm.
    projector : :class:`~.Projector`
        Projector which should be used for the projection of the 3-dimensional magnetization
        distribution.
    v_0 : float, optional
        The mean inner potential of the specimen in V. The default is 1.
    v_acc : float, optional
        The acceleration voltage of the electron microscope in V. The default is 30000.
    threshold : float, optional
        Threshold for the recognition of the specimen location. The default is 0, which means that
        all voxels with non-zero magnetization will contribute. Should be above noise level.

    '''

    _log = logging.getLogger(__name__+'.PhaseMapperElectric')

    def __init__(self, a, dim_uv, v_0=1, v_acc=30000, threshold=0):
        self._log.debug('Calling __init__')
        self.a = a
        self.dim_uv = dim_uv
        self.v_0 = v_0
        self.v_acc = v_acc
        self.threshold = threshold
        self.m = np.prod(self.dim_uv)
        self.n = np.prod(self.dim_uv)
        # Coefficient calculation:
        lam = H_BAR / np.sqrt(2 * M_E * Q_E * v_acc * (1 + Q_E*v_acc / (2*M_E*C**2)))
        C_e = 2*pi*Q_E/lam * (Q_E*v_acc + M_E*C**2) / (Q_E*v_acc * (Q_E*v_acc + 2*M_E*C**2))
        self.coeff = v_0 * C_e * a * 1E-9
        self._log.debug('Created '+str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(a=%r, dim_uv=%r, v_0=%r, v_acc=%r, threshold=%r)' % \
            (self.__class__, self.a, self.dim_uv, self.v_0, self.v_acc, self.threshold)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'PhaseMapperElectric(a=%s, dim_uv=%s, v_0=%s, v_acc=%s, threshold=%s)' % \
            (self.a, self.dim_uv, self.v_0, self.v_acc, self.threshold)

    def __call__(self, mag_data):
        self._log.debug('Calling __call__')
        assert isinstance(mag_data, MagData), 'Only MagData objects can be mapped!'
        assert mag_data.a == self.a, 'Grid spacing has to match!'
        assert mag_data.dim[0] == 1, 'Magnetic distribution must be 2-dimensional!'
        assert mag_data.dim[1:3] == self.dim_uv, 'Dimensions do not match!'
        return self.coeff * mag_data.get_mask(self.threshold)[0, ...].reshape(self.dim_uv)
        # Calculate mask:
        mask = mag_data.get_mask(self.threshold)
        # Project and calculate phase:
        # TODO: check if projector manages scalar multiplication (problem with MagData) and fix!
        projection = self.projector(mask.reshape(-1)).reshape(self.projector.dim_uv)
        phase = self.coeff * projection
        return PhaseMap(mag_data.a, phase)

    def jac_dot(self, vector):
        raise NotImplementedError()
        # TODO: Implement?

    def jac_T_dot(self, vector):
        raise NotImplementedError()
        # TODO: Implement?


def pm(mag_data, axis='z', dim_uv=None, b_0=1):
    '''Convenience function for fast phase mapping.

    Parameters
    ----------
    mag_data : :class:`~.MagData`
        A :class:`~.MagData` object, from which the projected phase map should be calculated.
    axis: {'z', 'y', 'x'}, optional
        Axis along which the :class:`.~SimpleProjector` projects the magnetic distribution.
    dim_uv : tuple of int (N=2), optional
        Dimensions of the 2-dimensional projected magnetization grid from which the phase should
        be calculated.
    b_0 : float, optional
        Saturation magnetization in Tesla, which is used for the phase calculation. Default is 1.
    Returns
    -------
    mag_data : :class:`~pyramid.magdata.MagData`
        The reconstructed magnetic distribution as a :class:`~.MagData` object.

    '''
    _log.debug('Calling pm')
    projector = SimpleProjector(mag_data.dim, axis=axis, dim_uv=dim_uv)
    phasemapper = PhaseMapperRDFC(Kernel(mag_data.a, projector.dim_uv))
    return phasemapper(projector(mag_data))

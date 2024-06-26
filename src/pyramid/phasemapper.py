# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module executes several forward models to calculate the magnetic or electric phase map from
a given projection of a 3-dimensional magnetic distribution (see :mod:`~pyramid.projector`).
For the magnetic phase map, an approach using real space and one using Fourier space is provided.
The electrostatic contribution is calculated by using the assumption of a mean inner potential."""

import abc
import logging

import numpy as np

from jutil import fft

from .fielddata import VectorData, ScalarData
from .phasemap import PhaseMap

__all__ = ['PhaseMapperRDFC', 'PhaseMapperFDFC', 'PhaseMapperMIP', 'PhaseMapperCharge']
_log = logging.getLogger(__name__)

PHI_0 = 2067.83  # magnetic flux in T*nm²
H_BAR = 6.626E-34  # Planck constant in J*s
M_E = 9.109E-31  # electron mass in kg
Q_E = 1.602E-19  # electron charge in C
C = 2.998E8  # speed of light in m/s
EPS_0 = 8.8542E-12  # electrical field constant


class PhaseMapper(object, metaclass=abc.ABCMeta):
    """Abstract base class for the phase calculation from a 2-dimensional distribution.

    The :class:`~.PhaseMapper-` class represents a strategy for the phasemapping of a
    2-dimensional magnetic/electric distribution onto a scalar phase map. :class:`~.PhaseMapper`
    is an abstract base class and provides a unified interface which should be subclassed with
    custom :func:`__init__` and :func:`__call__` functions. Concrete subclasses
    can be called as a function and take a :class:`~.FieldData` object as input and return a
    :class:`~.PhaseMap` object.

    """

    _log = logging.getLogger(__name__ + '.PhaseMapper')

    @abc.abstractmethod
    def __call__(self, field_data):
        raise NotImplementedError()

    @abc.abstractmethod
    def jac_dot(self, vector):
        """Calculate the product of the Jacobi matrix with a given `vector`.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized form of the field.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the Jacobi matrix (which is not explicitely calculated) with the vector.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def jac_T_dot(self, vector):
        """Calculate the product of the transposed Jacobi matrix with a given `vector`.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vector which represents a matrix with dimensions like a scalar phasemap.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the transposed Jacobi matrix (which is not explicitely calculated) with
            the vector.

        """
        raise NotImplementedError()


class PhaseMapperRDFC(PhaseMapper):
    """Class representing a phase mapping strategy using real space discretization and Fourier
    space convolution.

    The :class:`~.PMConvolve` class represents a phase mapping strategy involving discretization in
    real space. It utilizes the convolution in Fourier space, directly takes :class:`~.VectorData`
    objects and returns :class:`~.PhaseMap` objects.

    Attributes
    ----------
    kernel : :class:`~pyramid.Kernel`
        Convolution kernel, representing the phase contribution of one single magnetized pixel.
    m: int
        Size of the image space.
    n: int
        Size of the input space.

    """

    _log = logging.getLogger(__name__ + '.PhaseMapperRDFC')

    def __init__(self, kernel):
        self._log.debug('Calling __init__')
        self.kernel = kernel
        self.m = np.prod(kernel.dim_uv)
        self.n = 2 * self.m
        self.u_mag = np.zeros(kernel.dim_pad, dtype=kernel.u.dtype)
        self.v_mag = np.zeros(kernel.dim_pad, dtype=kernel.u.dtype)
        self.phase_adj = np.zeros(kernel.dim_pad, dtype=kernel.u.dtype)
        self._log.debug('Created ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(kernel=%r)' % (self.__class__, self.kernel)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'PhaseMapperRDFC(kernel=%s)' % self.kernel

    def __call__(self, magdata):
        assert isinstance(magdata, VectorData), 'Only VectorData objects can be mapped!'
        assert magdata.a == self.kernel.a, 'Grid spacing has to match!'
        assert magdata.dim[0] == 1, 'Magnetic distribution must be 2-dimensional!'
        assert magdata.dim[1:3] == self.kernel.dim_uv, 'Dimensions do not match!'
        # Process input parameters:
        self.u_mag[self.kernel.slice_mag] = magdata.field[0, 0, ...]  # u-component
        self.v_mag[self.kernel.slice_mag] = magdata.field[1, 0, ...]  # v-component
        return PhaseMap(magdata.a, self._convolve())

    def _convolve(self):
        # Fourier transform the projected magnetisation:
        self.u_mag_fft = fft.rfftn(self.u_mag)
        self.v_mag_fft = fft.rfftn(self.v_mag)
        # Convolve the magnetization with the kernel in Fourier space:
        self.phase_fft = self.u_mag_fft * self.kernel.u_fft + self.v_mag_fft * self.kernel.v_fft
        # Return the result:
        return fft.irfftn(self.phase_fft)[self.kernel.slice_phase]

    def jac_dot(self, vector):
        """Calculate the product of the Jacobi matrix with a given `vector`.

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

        """
        assert len(vector) == self.n, \
            'vector size not compatible! vector: {}, size: {}'.format(len(vector), self.n)
        self.u_mag[self.kernel.slice_mag], self.v_mag[self.kernel.slice_mag] = \
            np.reshape(vector, (2,) + self.kernel.dim_uv)
        return np.ravel(self._convolve())

    def jac_T_dot(self, vector):
        """Calculate the product of the transposed Jacobi matrix with a given `vector`.

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

        """
        assert len(vector) == self.m, \
            'vector size not compatible! vector: {}, size: {}'.format(len(vector), self.m)
        self.phase_adj[self.kernel.slice_phase] = vector.reshape(self.kernel.dim_uv)
        phase_adj_fft = fft.irfft2_adj(self.phase_adj)
        u_mag_adj_fft = phase_adj_fft * np.conj(self.kernel.u_fft)
        v_mag_adj_fft = phase_adj_fft * np.conj(self.kernel.v_fft)
        u_mag_adj = fft.rfft2_adj(u_mag_adj_fft)[self.kernel.slice_mag]
        v_mag_adj = fft.rfft2_adj(v_mag_adj_fft)[self.kernel.slice_mag]
        result = np.concatenate((u_mag_adj.ravel(), v_mag_adj.ravel()))
        return result


class PhaseMapperFDFC(PhaseMapper):
    """Class representing a phase mapping strategy using a discretization in Fourier space.

    The :class:`~.PMFourier` class represents a phase mapping strategy involving discretization in
    Fourier space. It utilizes the Fourier transforms, which are inherently calculated in the
    :class:`~.Kernel` class and directly takes :class:`~.VectorData` objects and returns
    :class:`~.PhaseMap` objects.

    Attributes
    ----------
    a : float
        The grid spacing in nm.
    dim_uv : tuple of int (N=2)
        Dimensions of the 2-dimensional projected magnetization grid for the kernel setup.
    b_0 : float, optional
        The magnetic induction corresponding to a magnetization :math:`M_{0}` in T.
        The default is 1.
    padding : int, optional
        Factor for the zero padding. The default is 0 (no padding). For a factor of n the number
        of pixels is increase by ``(1+n)**2``. Padded zeros are cropped at the end.
    m: int
        Size of the image space.
    n: int
        Size of the input space.

    """

    _log = logging.getLogger(__name__ + '.PhaseMapperFDFC')

    def __init__(self, a, dim_uv, b_0=1, padding=0):
        self._log.debug('Calling __init__')
        self.a = a
        self.dim_uv = dim_uv
        self.b_0 = b_0
        self.padding = padding
        self.m = np.prod(dim_uv)
        self.n = 2 * self.m
        self._log.debug('Created ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(a=%r, dim_uv=%r, b_0=%r, padding=%r)' % \
               (self.__class__, self.a, self.dim_uv, self.b_0, self.padding)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'PhaseMapperFDFC(a=%s, dim_uv=%s, b_0=%s, padding=%s)' % \
               (self.a, self.dim_uv, self.b_0, self.padding)

    def __call__(self, magdata):
        self._log.debug('Calling __call__')
        assert isinstance(magdata, VectorData), 'Only VectorData objects can be mapped!'
        assert magdata.a == self.a, 'Grid spacing has to match!'
        assert magdata.dim[0] == 1, 'Magnetic distribution must be 2-dimensional!'
        assert magdata.dim[1:3] == self.dim_uv, 'Dimensions do not match!'
        v_dim, u_dim = self.dim_uv
        u_mag, v_mag = magdata.field[0:2, 0, ...]
        # Create zero padded matrices:
        u_pad = int(u_dim / 2 * self.padding)
        v_pad = int(v_dim / 2 * self.padding)
        u_mag_pad = np.pad(u_mag, ((v_pad, v_pad), (u_pad, u_pad)), 'constant')
        v_mag_pad = np.pad(v_mag, ((v_pad, v_pad), (u_pad, u_pad)), 'constant')
        # Fourier transform of the two components:
        u_mag_fft = np.fft.rfft2(u_mag_pad)
        v_mag_fft = np.fft.rfft2(v_mag_pad)
        # Calculate the Fourier transform of the phase:
        f_u = np.fft.rfftfreq(u_dim + 2 * u_pad, self.a)
        f_v = np.fft.fftfreq(v_dim + 2 * v_pad, self.a)
        f_uu, f_vv = np.meshgrid(f_u, f_v)
        coeff = - (1j * self.b_0 * self.a) / (2 * PHI_0)  # Minus because of negative z-direction
        phase_fft = coeff * (u_mag_fft * f_vv - v_mag_fft * f_uu) / (f_uu ** 2 + f_vv ** 2 + 1e-30)
        # Transform to real space and revert padding:
        phase_pad = np.fft.irfft2(phase_fft)
        phase = phase_pad[v_pad:v_pad + v_dim, u_pad:u_pad + u_dim]
        return PhaseMap(magdata.a, phase)

    def jac_dot(self, vector):
        """Calculate the product of the Jacobi matrix with a given `vector`.

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

        """
        self._log.debug('Calling jac_dot')
        assert len(vector) == self.n, \
            'vector size not compatible! vector: {}, size: {}'.format(len(vector), self.n)
        mag_proj = VectorData(self.a, np.zeros((3, 1) + self.dim_uv, dtype=np.float32))
        magnitude_proj = np.reshape(vector, (2,) + self.dim_uv)
        mag_proj.field[:2, 0, ...] = magnitude_proj
        return self(mag_proj).phase_vec

    def jac_T_dot(self, vector):
        """Calculate the product of the transposed Jacobi matrix with a given `vector`.

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

        """
        raise NotImplementedError()


class PhaseMapperMIP(PhaseMapper):
    """Class representing a phase mapping strategy for the electrostatic contribution.

    The :class:`~.PhaseMapperMIP` class represents a phase mapping strategy for the electrostatic
    contribution to the electron phase shift which results e.g. from the mean inner potential in
    certain samples and which is sensitive to properties of the electron microscope. It directly
    takes :class:`~.ScalarData` objects and returns :class:`~.PhaseMap` objects.

    Attributes
    ----------
    a : float
        The grid spacing in nm.
    dim_uv : tuple of int (N=2)
        Dimensions of the 2-dimensional projected magnetization grid for the kernel setup.
    v_0 : float, optional
        The mean inner potential of the specimen in V. The default is 1.
    v_acc : float, optional
        The acceleration voltage of the electron microscope in V. The default is 300000.
    threshold : float, optional
        Threshold for the recognition of the specimen location. The default is 0, which means that
        all voxels with non-zero magnetization will contribute. Should be above noise level.
    m: int
        Size of the image space.
    n: int
        Size of the input space.

    """

    _log = logging.getLogger(__name__ + '.PhaseMapperMIP')

    def __init__(self, a, dim_uv, v_0=1, v_acc=30000, threshold=0):
        self._log.debug('Calling __init__')
        self.a = a
        self.dim_uv = dim_uv
        self.v_0 = v_0
        self.v_acc = v_acc
        self.threshold = threshold
        self.m = np.prod(self.dim_uv)
        self.n = self.m
        # Coefficient calculation:
        lam = H_BAR / np.sqrt(2 * M_E * Q_E * v_acc * (1 + Q_E * v_acc / (2 * M_E * C ** 2)))
        C_e = 2 * np.pi * Q_E / lam * (Q_E * v_acc + M_E * C ** 2) / (
            Q_E * v_acc * (Q_E * v_acc + 2 * M_E * C ** 2))
        self.coeff = v_0 * C_e * a * 1E-9
        self._log.debug('Created ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(a=%r, dim_uv=%r, v_0=%r, v_acc=%r, threshold=%r)' % \
               (self.__class__, self.a, self.dim_uv, self.v_0, self.v_acc, self.threshold)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'PhaseMapperMIP(a=%s, dim_uv=%s, v_0=%s, v_acc=%s, threshold=%s)' % \
               (self.a, self.dim_uv, self.v_0, self.v_acc, self.threshold)

    def __call__(self, elec_data):
        self._log.debug('Calling __call__')
        assert isinstance(elec_data, ScalarData), 'Only ScalarData objects can be mapped!'
        assert elec_data.a == self.a, 'Grid spacing has to match!'
        assert elec_data.dim[0] == 1, 'Magnetic distribution must be 2-dimensional!'
        assert elec_data.dim[1:3] == self.dim_uv, 'Dimensions do not match!'
        phase = self.coeff * np.squeeze(elec_data.get_mask(self.threshold))
        return PhaseMap(elec_data.a, phase)

    def jac_dot(self, vector):
        """Calculate the product of the Jacobi matrix with a given `vector`.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized form of the electrostatic field of every pixel (row-wise).

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the Jacobi matrix (which is not explicitely calculated) with the vector.

        """
        raise NotImplementedError()  # TODO: Implement right!

    def jac_T_dot(self, vector):
        """Calculate the product of the transposed Jacobi matrix with a given `vector`.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vector with ``N**2`` entries which represents a matrix with dimensions like a scalar
            phasemap.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the transposed Jacobi matrix (which is not explicitly calculated) with
            the vector, which has ``N**2`` entries like an electrostatic projection.

        """
        raise NotImplementedError()  # TODO: Implement right!


class PhaseMapperCharge(PhaseMapper):
    """Class representing a phase mapping strategy for the electrostatic charge contribution.

    The :class:`~.PhaseMapperCharge` class represents a phase mapping strategy for the electrostatic
    charge contribution to the electron phase shift which results e.g. from the charges in
    certain samples and which is sensitive to properties of the electron microscope. It directly
    takes :class:`~.ScalarData` objects and returns :class:`~.PhaseMap` objects.

    Attributes
    ----------
    kernelcharge : :class:`~pyramid.KernelCharge`
        Convolution kernel, representing the phase contribution of one single charge pixel.
    m: int
        Size of the image space.
    n: int
        Size of the input space.
    c: :class:`~numpy.ndarray` (N=3)
        The charge distribution
    c_fft : :class:`~numpy.ndarray` (N=3)
        The real FFT of the charge distribution.
    phase_fft: :class:`~numpy.ndarray` (N=3)
        The real FFT of the phase from the given charge distribution c.

    """
    _log = logging.getLogger(__name__ + '.PhaseMapperCharge')

    def __init__(self, kernelcharge):
        self._log.debug('Calling __init__')
        self.kernelcharge = kernelcharge
        self.m = np.prod(kernelcharge.dim_uv)
        self.n = self.m
        self.c = np.zeros(kernelcharge.dim_pad, dtype=kernelcharge.kc.dtype)
        self.phase_adj = np.zeros(kernelcharge.dim_pad, dtype=kernelcharge.kc.dtype)
        self._log.debug('Created ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(kernelcharge=%r)' % (self.__class__, self.kernelcharge)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'PhaseMapperCharge(kernelcharge=%s)' % self.kernelcharge

    def __call__(self, elecdata):
        assert isinstance(elecdata, ScalarData), 'Only ScalarData objects can be mapped!'
        assert elecdata.a == self.kernelcharge.a, 'Grid spacing has to match!'
        assert elecdata.dim[0] == 1, 'Charge distribution must be 2-dimensional!'
        assert elecdata.dim[1:3] == self.kernelcharge.dim_uv, 'Dimensions do not match!'
        # Process input parameters:
        self.c[self.kernelcharge.slice_c] = elecdata.field[0, ...]
        return PhaseMap(elecdata.a, self._convolve())

    def _convolve(self):
        # Fourier transform of the projected charge distribution:
        self.c_fft = fft.rfftn(self.c)
        # Convolve the charge distribution with the kernel in Fourier space:
        self.phase_fft = self.c_fft * self.kernelcharge.kc_fft
        # Return the result:
        return fft.irfftn(self.phase_fft)[self.kernelcharge.slice_phase]

    def jac_dot(self, vector):
        """Calculate the product of the Jacobi matrix with a given `vector`.

        Parameters
        ----------
        vector: :class:`~numpy.ndarray` (N=1)
            Vectorized form of the charge distribution of every pixel (row-wise).

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the Jacobi matrix (which is not explicitly calculated) with the vector.

        """
        assert len(vector) == self.n, \
            'vector size not compatible! vector: {}, size: {}'.format(len(vector), self.n)
        self.c[self.kernelcharge.slice_c] = np.reshape(vector, self.kernelcharge.dim_uv)
        return np.ravel(self._convolve())

    def jac_T_dot(self, vector):
        """Calculate the product of the transposed Jacobi matrix with a given `vector`.

        Parameters
        ----------
        vector: :class:`~numpy.ndarray` (N=1)
            Vector with ``N**2`` entries which represents a matrix with dimensions like a vector
            phasemap.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the transposed Jacobi matrix (which is not explicitly calculated) with
            the vector, which has ``N**2`` entries like a 2D charge projection.

        """
        assert len(vector) == self.m, \
            'vector size not compatible! vector: {}, size: {}'.format(len(vector), self.m)
        self.phase_adj[self.kernelcharge.slice_phase] = vector.reshape(self.kernelcharge.dim_uv)
        phase_adj_fft = fft.irfft2_adj(self.phase_adj)
        kc_adj_fft = phase_adj_fft * np.conj(self.kernelcharge.kc_fft)
        kc_adj = fft.rfft2_adj(kc_adj_fft)[self.kernelcharge.slice_c]
        result = kc_adj.ravel()
        return result

# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.KernelCharge` class, representing the phase contribution of one
single Charged pixel."""

import logging

import numpy as np

from jutil import fft

__all__ = ['Kernel', 'PHI_0'] # TODO rewrite!

# PHI_0 = 2067.83  # magnetic flux in T*nm²
H_BAR = 6.626E-34  # Planck constant in J*s
M_E = 9.109E-31  # electron mass in kg
Q_E = 1.602E-19  # electron charge in C
C = 2.998E8  # speed of light in m/s
EPS_0 = 8.8542E-12  # electrical field constant


class KernelCharge(object):
    """Class for calculating kernel matrices for the phase calculation.

    Represents the phase of a single charged pixel, which can be accessed via the corresponding attributes.
    During the construction, a few attributes are calculated that are used in
    the convolution during phase calculation in the different :class:`~Phasemapper` classes.

    Attributes
    ----------
    a : float
        The grid spacing in nm.
    v_acc : float, optional
            The acceleration voltage of the electron microscope in V. The default is 300000.
    electrode_vec : tuple of float (N=2)
        The norm vector of the counter electrode, (elec_a,elec_b), and the distance to the origin is
        the norm of (elec_a,elec_b).
    dim_uv : tuple of int (N=2), optional
        Dimensions of the 2-dimensional projected magnetization grid from which the phase should
        be calculated.
    dim_kern : tuple of int (N=2)
        Dimensions of the kernel, which is ``2N-1`` for both axes compared to `dim_uv`.
    dim_pad : tuple of int (N=2)
        Dimensions of the padded FOV, which is ``2N`` (if FFTW is used) or the next highest power
        of 2 (for numpy-FFT).
    dim_fft : tuple of int (N=2)
        Dimensions of the grid, which is used for the FFT, taking into account that a RFFT should
        be used (one axis is halved in comparison to `dim_pad`).
    kc : :class:`~numpy.ndarray` (N=3)
        The phase contribution of one charged pixel.
    kc_fft : :class:`~numpy.ndarray` (N=3)
        The real FFT of the phase contribution of one charged pixel.
    slice_phase : tuple (N=2) of :class:`slice`
        A tuple of :class:`slice` objects to extract the original FOV from the increased one with
        size `dim_pad` for the elementary kernel phase. The kernel is shifted, thus the center is
        not at (0, 0), which also shifts the slicing compared to `slice_mag`.
    slice_mag : tuple (N=2) of :class:`slice`
        A tuple of :class:`slice` objects to extract the original FOV from the increased one with
        size `dim_pad` for the projected magnetization distribution.
    prw_vec: tuple of 2 int, optional
        A two-component vector describing the displacement of the reference wave to include
        perturbation of this reference by the object itself (via fringing fields), (y, x).
    dtype: numpy dtype, optional
        Data type of the kernel. Default is np.float32.

    """

    _log = logging.getLogger(__name__ + '.Kernel')  # TODO 'KernelCharge'

    def __init__(self, a, dim_uv, electrode_vec, v_acc=300000., prw_vec=None, dtype=np.float32):
        self._log.debug('Calling __init__')
        # Set basic properties:
        self.prw_vec = prw_vec
        self.dim_uv = dim_uv  # Dimensions of the FOV
        self.dim_kern = tuple(2 * np.array(dim_uv) - 1)  # Dimensions of the kernel
        self.a = a
        self.electrode_vec = electrode_vec
        self.v_acc = v_acc
        lam = H_BAR / np.sqrt(2 * M_E * Q_E * v_acc * (1 + Q_E * v_acc / (2 * M_E * C ** 2)))
        c_e = 2 * np.pi * Q_E / lam * (Q_E * v_acc + M_E * C ** 2) / (
            Q_E * v_acc * (Q_E * v_acc + 2 * M_E * C ** 2))

        # Set up FFT:
        if fft.HAVE_FFTW:

            self.dim_pad = tuple(2 * np.array(dim_uv))  # is at least even (not nec. power of 2)
        else:
            self.dim_pad = tuple(2 ** np.ceil(np.log2(2 * np.array(dim_uv))).astype(int))  # pow(2)

        self.dim_fft = (self.dim_pad[0], self.dim_pad[1] // 2 + 1)  # last axis is real
        self.slice_phase = (slice(dim_uv[0] - 1, self.dim_kern[0]),  # Shift because kernel center
                            slice(dim_uv[1] - 1, self.dim_kern[1]))  # is not at (0, 0)!
        self.slice_mag = (slice(0, dim_uv[0]),  # Magnetization is padded on the far end!
                          slice(0, dim_uv[1]))  # (Phase cutout is shifted as listed above)
        # Calculate kernel (single pixel phase):
        coeff = c_e * Q_E / (4 * np.pi * EPS_0)  # Minus is gone because of negative z-direction
        v_dim, u_dim = dim_uv
        u = np.linspace(-(u_dim - 1), u_dim - 1, num=2 * u_dim - 1)
        v = np.linspace(-(v_dim - 1), v_dim - 1, num=2 * v_dim - 1)
        uu, vv = np.meshgrid(u, v)
        self.kc = np.empty(self.dim_kern, dtype=dtype)
        self.kc[...] = coeff * self._get_elementary_phase(electrode_vec, uu, vv, a)
        # Include perturbed reference wave:
        if prw_vec is not None:
            uu += prw_vec[1]
            vv += prw_vec[0]
            self.kc[...] -= coeff * self._get_elementary_phase(electrode_vec, uu, vv, a)
        # Calculate Fourier transform of kernel:
        self.kc_fft = fft.rfftn(self.kc, self.dim_pad)
        self._log.debug('Created ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(a=%r, dim_uv=%r, electrode_vec=%r,prw_vec=%r)' % \
               (self.__class__, self.a, self.dim_uv, self.electrode_vec, self.prw_vec)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'Kernel(a=%s, dim_uv=%s, electrode_vec=%s, prw_vec=%s,)' % \
               (self.a, self.dim_uv, self.electrode_vec, self.prw_vec)

    def _get_elementary_phase(self, electrode_vec, n, m, a):
        self._log.debug('Calling _get_elementary_phase')
        elec_a, elec_b = electrode_vec
        n_img = 2 * elec_a
        m_img = 2 * elec_b
        in_or_out1 = ~ np.logical_and(n == 0, m == 0)
        in_or_out2 = ~ np.logical_and((n - n_img) == 0, (m - m_img) == 0)
        return (1. / np.sqrt(a ** 2 * (n ** 2 + m ** 2 + 1E-30))) * in_or_out1 - \
               (1. / np.sqrt(a ** 2 * ((n - n_img) ** 2 + (m - m_img) ** 2 + 1E-30))) * in_or_out2

    def print_info(self):
        """Print information about the kernel.

        Returns
        -------
        None

        """
        self._log.debug('Calling log_info')
        print('Shape of the FOV    :', self.dim_uv)
        print('Shape of the Kernel :', self.dim_kern)
        print('Zero-padded shape   :', self.dim_pad)
        print('Shape of the FFT    :', self.dim_fft)
        print('Slice for the phase :', self.slice_phase)
        print('Slice for the magn. :', self.slice_mag)
        print('Grid spacing        : {} nm'.format(self.a))
        print('PRW vector          : {} T'.format(self.prw_vec))
        print('Electrode vector    : {} T'.format(self.electrode_vec))

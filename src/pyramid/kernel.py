# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.Kernel` class, representing the phase contribution of one
single magnetized pixel."""

import logging

import numpy as np

from jutil import fft

__all__ = ['Kernel', 'PHI_0', 'KernelCharge']

PHI_0 = 2067.83  # magnetic flux in T*nm²
H_BAR = 6.626E-34  # Planck constant in J*s
M_E = 9.109E-31  # electron mass in kg
Q_E = 1.602E-19  # electron charge in C
C = 2.998E8  # speed of light in m/s
EPS_0 = 8.8542E-12  # electrical field constant


class Kernel(object):
    """Class for calculating kernel matrices for the phase calculation.

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
    dim_kern : tuple of int (N=2)
        Dimensions of the kernel, which is ``2N-1`` for both axes compared to `dim_uv`.
    dim_pad : tuple of int (N=2)
        Dimensions of the padded FOV, which is ``2N`` (if FFTW is used) or the next highest power
        of 2 (for numpy-FFT).
    dim_fft : tuple of int (N=2)
        Dimensions of the grid, which is used for the FFT, taking into account that a RFFT should
        be used (one axis is halved in comparison to `dim_pad`).
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

    _log = logging.getLogger(__name__ + '.Kernel')

    def __init__(self, a, dim_uv, b_0=1., prw_vec=None, geometry='disc', dtype=np.float32):
        self._log.debug('Calling __init__')
        # Set basic properties:
        self.b_0 = b_0
        self.prw_vec = prw_vec
        self.dim_uv = dim_uv  # Dimensions of the FOV
        self.dim_kern = tuple(2 * np.array(dim_uv) - 1)  # Dimensions of the kernel
        self.a = a
        self.geometry = geometry
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
        # [M_0] = A/m  --> This is the magnetization, not the magnetic moment (A/m * m³ = Am²)!
        # [PHI_0 / µ_0] = Tm² / Tm/A = Am
        # [b_0] = [M_0] * [µ_0] = A/m * N/A² = N/Am = T
        # [coeff] = [b_0 * a² / (2*PHI_0)] = T * m² / Tm² = 1  --> without unit (phase)!
        coeff = b_0 * a ** 2 / (2 * PHI_0)  # Minus is gone because of negative z-direction
        v_dim, u_dim = dim_uv
        u = np.linspace(-(u_dim - 1), u_dim - 1, num=2 * u_dim - 1)
        v = np.linspace(-(v_dim - 1), v_dim - 1, num=2 * v_dim - 1)
        uu, vv = np.meshgrid(u, v)
        # TODO: u, v are coordinates, rename self.u/v to self.kern_u/v!
        self.u = np.empty(self.dim_kern, dtype=dtype)  # TODO: Is this necessary?
        self.v = np.empty(self.dim_kern, dtype=dtype)  # TODO: Move to _get_elementary_phase?
        self.u[...] = coeff * self._get_elementary_phase(geometry, uu, vv, a)
        # TODO: The minus sign belongs into the phasemapper (debatable)!
        self.v[...] = coeff * -self._get_elementary_phase(geometry, vv, uu, a)
        # Include perturbed reference wave:
        if prw_vec is not None:
            uu += prw_vec[1]
            vv += prw_vec[0]
            self.u[...] -= coeff * self._get_elementary_phase(geometry, uu, vv, a)
            # TODO: The minus sign belongs into the phasemapper (debatable)!
            self.v[...] -= coeff * -self._get_elementary_phase(geometry, vv, uu, a)
        # Calculate Fourier trafo of kernel components:
        self.u_fft = fft.rfftn(self.u, self.dim_pad)
        self.v_fft = fft.rfftn(self.v, self.dim_pad)
        self._log.debug('Created ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(a=%r, dim_uv=%r, b_0=%r, prw_vec=%r, geometry=%r)' % \
               (self.__class__, self.a, self.dim_uv, self.b_0, self.prw_vec, self.geometry)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'Kernel(a=%s, dim_uv=%s, b_0=%s, prw_vec=%s, geometry=%s)' % \
               (self.a, self.dim_uv, self.b_0, self.prw_vec, self.geometry)

    def _get_elementary_phase(self, geometry, n, m, a):
        self._log.debug('Calling _get_elementary_phase')
        # TODO: Rename n m to p q ?
        if geometry == 'disc':
            in_or_out = ~ np.logical_and(n == 0, m == 0)
            return m / (n ** 2 + m ** 2 + 1E-30) * in_or_out
        elif geometry == 'slab':
            def _F_a(n, m):
                A = np.log(a ** 2 * (n ** 2 + m ** 2))
                B = np.arctan(n / m)
                return n * A - 2 * n + 2 * m * B

            return 0.5 * (_F_a(n - 0.5, m - 0.5) - _F_a(n + 0.5, m - 0.5)
                          - _F_a(n - 0.5, m + 0.5) + _F_a(n + 0.5, m + 0.5))

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
        print('Saturation Induction:', self.b_0)
        print('Grid spacing        : {} nm'.format(self.a))
        print('Geometry            :', self.geometry)
        print('PRW vector          : {} T'.format(self.prw_vec))


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
        The norm vector of the counter electrode in pixels, (elec_a,elec_b), and the distance to the origin is
        the norm of (elec_a,elec_b).
    dim_uv : tuple of int (N=2), optional
        Dimensions of the 2-dimensional electrostatic charge grid from which the phase should
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
        not at (0, 0), which also shifts the slicing compared to `slice_c`.
    slice_c : tuple (N=2) of :class:`slice`
        A tuple of :class:`slice` objects to extract the original FOV from the increased one with
        size `dim_pad` for the projected charge distribution.
    prw_vec: tuple of 2 int, optional
        A two-component vector describing the displacement of the reference wave to include
        perturbation of this reference by the object itself (via fringing fields), (y, x).
    dtype: numpy dtype, optional
        Data type of the kernel. Default is np.float32.

    """

    _log = logging.getLogger(__name__ + '.KernelCharge')

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
        C_e = 2 * np.pi * Q_E / lam * (Q_E * v_acc + M_E * C ** 2) / (
            Q_E * v_acc * (Q_E * v_acc + 2 * M_E * C ** 2))
        # Set up FFT:
        if fft.HAVE_FFTW:
            self.dim_pad = tuple(2 * np.array(dim_uv))  # is at least even (not nec. power of 2)
        else:
            self.dim_pad = tuple(2 ** np.ceil(np.log2(2 * np.array(dim_uv))).astype(int))  # pow(2)
        self.dim_fft = (self.dim_pad[0], self.dim_pad[1] // 2 + 1)  # last axis is real
        self.slice_phase = (slice(dim_uv[0] - 1, self.dim_kern[0]),  # Shift because kernel center
                            slice(dim_uv[1] - 1, self.dim_kern[1]))  # is not at (0, 0)!
        self.slice_c = (slice(0, dim_uv[0]),  # Charge is padded on the far end!
                        slice(0, dim_uv[1]))  # (Phase cutout is shifted as listed above)
        # Calculate kernel (single pixel phase):
        coeff = a * C_e * Q_E / (4 * np.pi * EPS_0)  # Minus gone because of negative z-direction
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
        return '%s(a=%r, dim_uv=%r, electrode_vec=%r, prw_vec=%r)' % \
               (self.__class__, self.a, self.dim_uv, self.electrode_vec, self.prw_vec)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'Kernel(a=%s, dim_uv=%s, electrode_vec=%s, prw_vec=%s,)' % \
               (self.a, self.dim_uv, self.electrode_vec, self.prw_vec)

    def _get_elementary_phase(self, electrode_vec, n, m, a):
        self._log.debug('Calling _get_elementary_phase')
        elec_a, elec_b = electrode_vec
        # The positions of the image charge, i.e., 2 * electrode_vec as the single charge sits at (0, 0)
        u_img = 2 * elec_a
        v_img = 2 * elec_b
        # The projected distance from the charges or image charges
        r1 = np.sqrt(n ** 2 + m ** 2)
        r2 = np.sqrt((n - u_img) ** 2 + (m - v_img) ** 2)
        # The square height when  the path come across the sphere
        R = 1. / 2  # the radius of the pixel
        h1 = R ** 2 - r1 ** 2
        h2 = R ** 2 - r2 ** 2
        # Phase calculation in 3 different cases
        # case 1 totally out of the sphere
        case1 = ((h1 < 0) & (h2 < 0))
        phase = np.zeros(n.shape)
        # elec_field = np.zeros(n.shape)
        phase[case1] += - np.log((r1[case1] ** 2) / (r2[case1] ** 2))
        # The x, y component of electric field, Ex=Ey in the kernel
        # elec_field[case1] += np.sqrt(3) * np.pi * (1 / r1[case1] - 1 / r2[case1]) / 3
        # case 2: inside the charge sphere
        case2 = ((h1 >= 0) & (h2 <= 0))
        # The height when the path come across the charge sphere
        h3 = np.sqrt(np.where(h1 > 0, h1, 0))
        phase[case2] += (- np.log((h3[case2] + R) ** 2 / r2[case2] ** 2)
                         + (2 * h3[case2] / R + 2 * h3[case2] ** 3 / 3 / R ** 3))
        # elec_field[case2] += 2 * np.sqrt(3) / 3 * (np.pi * (1 / r1[case1] - 1 / r2[case1]) / 2 -
        #                                   np.arctan(h3[case2] / r1[case2]) / r1[case2] + h3[case2] / 2 / R ** 2 +
        #                                   r1[case2] ** 2 / 2 / R ** 3 * (np.log(h3[case2] + R)) - np.log(r1[case2]))
        # case 3 : inside the image charge sphere
        case3 = np.logical_not(case1 | case2)
        # The height whe the path comes across the image charge sphere
        h4 = np.sqrt(np.where(h2 > 0, h2, 0))
        phase[case3] += (np.log((h4[case3] + R) ** 2 / r1[case3] ** 2)
                         - (2 * h4[case3] / R + 2 * h4[case3] ** 3 / 3 / R ** 3))
        # elec_field[case3] += 2 * np.sqrt(3) / 3 * (np.pi * (1 / r1[case1] - 1 / r2[case1]) / 2 -
        #                                   np.arctan(h4[case3] / r2[case3]) / r2[case3] - h4[case3] / 2 / R ** 2 -
        #                                   r2[case3] ** 2 / 2 / R ** 3 * (np.log(h4[case3] + R)) - np.log(r2[case3]))
        return phase

    def print_info(self):
        """Print information about the kernel.

        Returns
        -------
        None

        """
        self._log.debug('Calling log_info')
        print('Shape of the FOV     :', self.dim_uv)
        print('Shape of the Kernel  :', self.dim_kern)
        print('Zero-padded shape    :', self.dim_pad)
        print('Shape of the FFT     :', self.dim_fft)
        print('Slice for the phase  :', self.slice_phase)
        print('Slice for the charge :', self.slice_c)
        print('Grid spacing         : {} nm'.format(self.a))
        print('PRW vector           : {} T'.format(self.prw_vec))
        print('Electrode vector     : {} T'.format(self.electrode_vec))
        print('Acceleration voltage : {} V'.format(self.v_acc))

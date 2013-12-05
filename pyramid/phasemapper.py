# -*- coding: utf-8 -*-
"""Create magnetic and electric phase maps from magnetization data.

This module executes several forward models to calculate the magnetic or electric phase map from
a given projection of a 3-dimensional magnetic distribution (see :mod:`~pyramid.projector`).
For the magnetic phase map, an approach using real space and one using Fourier space is provided.
The electrostatic contribution is calculated by using the assumption of a mean inner potential.

"""


import numpy as np
from numpy import pi

import pyramid.numcore as nc
from pyramid.kernel import Kernel


PHI_0 = -2067.83    # magnetic flux in T*nm²
H_BAR = 6.626E-34  # Planck constant in J*s
M_E = 9.109E-31    # electron mass in kg
Q_E = 1.602E-19    # electron charge in C
C = 2.998E8        # speed of light in m/s


def phase_mag(a, projection, b_0=1, kernel=None):
    '''Calculate the magnetic phase from magnetization data.

    Parameters
    ----------
    a : float
        The grid spacing in nm.
    projection : tuple (N=3) of :class:`~numpy.ndarray` (N=2)
        The in-plane projection of the magnetization as a tuple, storing the `u`- and `v`-component
        of the magnetization and the thickness projection for the resulting 2D-grid.
    b_0 : float, optional
        The magnetic induction corresponding to a magnetization `M`\ :sub:`0` in T.
        The default is 1.
    kernel : :class:`~pyramid.kernel.Kernel`, optional
        Specifies the kernel for the convolution with the magnetization data. If none is specified,
        one will be created with `disc` as the default geometry.

    Returns
    -------
    phase : :class:`~numpy.ndarray` (N=2)
        The phase as a 2-dimensional array.

    '''
    # Process input parameters:
    v_mag, u_mag = projection[:-1]
    dim = np.shape(u_mag)

    # Create kernel (lookup-tables for the phase of one pixel) if none is given:
    if kernel is None:
        kernel = Kernel(dim, a, b_0)

    # Fourier transform the projected magnetisation:
    u_mag_fft = np.fft.rfftn(u_mag, kernel.dim_fft)
    v_mag_fft = np.fft.rfftn(v_mag, kernel.dim_fft)

    # Convolve the magnetization with the kernel in Fourier space:
    u_phase = np.fft.irfftn(u_mag_fft * kernel.u_fft, kernel.dim_fft)[kernel.slice_fft].copy()
    v_phase = np.fft.irfftn(v_mag_fft * kernel.v_fft, kernel.dim_fft)[kernel.slice_fft].copy()

    # Return the result:
    return u_phase - v_phase


def phase_elec(a, projection, v_0=1, v_acc=30000):
    '''Calculate the electric phase from magnetization distributions.

    Parameters
    ----------
    a : float
        The grid spacing in nm.
    projection : tuple (N=3) of :class:`~numpy.ndarray` (N=2)
        The in-plane projection of the magnetization as a tuple, storing the u- and v-component
        of the magnetization and the thickness projection for the resulting 2D-grid.
    v_0 : float, optional
        The mean inner potential of the specimen in V. The default is 1.
    v_acc : float, optional
        The acceleration voltage of the electron microscope in V. The default is 30000.

    Returns
    -------
    phase : :class:`~numpy.ndarray` (N=2)
        The phase as a 2-dimensional array.

    '''
    # Process input parameters:
    lam = H_BAR / np.sqrt(2 * M_E * Q_E * v_acc * (1 + Q_E*v_acc / (2*M_E*C**2)))
    Ce = 2*pi*Q_E/lam * (Q_E*v_acc + M_E*C**2) / (Q_E*v_acc * (Q_E*v_acc + 2*M_E*C**2))
    # return phase:
    return v_0 * Ce * projection[-1] * a*1E-9


def phase_mag_real(a, projection, b_0=1, geometry='disc', jacobi=None):
    '''Calculate the magnetic phase from magnetization data (pure real space, no FFT-convolution).

    Parameters
    ----------
    a : float
        The grid spacing in nm.
    projection : tuple (N=3) of :class:`~numpy.ndarray` (N=2)
        The in-plane projection of the magnetization as a tuple, storing the `u`- and `v`-component
        of the magnetization and the thickness projection for the resulting 2D-grid.
    geometry : {'disc', 'slab'}, optional
        Specifies the elemental geometry to use for the pixel field.
        The default is 'disc', because of the smaller computational overhead.
    b_0 : float, optional
        The magnetic induction corresponding to a magnetization `M`\ :sub:`0` in T.
        The default is 1.
    jacobi : :class:`~numpy.ndarray` (N=2), optional
        Specifies the matrix in which to save the jacobi matrix. The jacobi matrix will not be
        calculated, if no matrix is specified (default), resulting in a faster computation.

    Returns
    -------
    phase : :class:`~numpy.ndarray` (N=2)
        The phase as a 2-dimensional array.

    '''
    # Process input parameters: 
    dim = np.shape(projection[0])
    v_mag, u_mag = projection[:-1]

    # Create kernel (lookup-tables for the phase of one pixel):
    kernel = Kernel(dim, a, b_0, geometry)
    u_phi = kernel.u
    v_phi = kernel.v

    # Calculation of the phase:
    phase = np.zeros(dim)
    threshold = 0
    if jacobi is not None:  # With Jacobian matrix (slower)
        jacobi[:] = 0  # Jacobi matrix --> zeros
        for j in range(dim[0]):
            for i in range(dim[1]):
                u_phase = u_phi[dim[0]-1-j:(2*dim[0]-1)-j, dim[1]-1-i:(2*dim[1]-1)-i]
                jacobi[:, i+dim[1]*j] = u_phase.reshape(-1)
                if abs(u_mag[j, i]) > threshold:
                    phase += u_mag[j, i] * u_phase
                v_phase = v_phi[dim[0]-1-j:(2*dim[0]-1)-j, dim[1]-1-i:(2*dim[1]-1)-i]
                jacobi[:, dim[1]*dim[0]+i+dim[1]*j] = -v_phase.reshape(-1)
                if abs(v_mag[j, i]) > threshold:
                    phase -= v_mag[j, i] * v_phase
    else:  # Without Jacobi matrix (faster)
        nc.phase_mag_real_core(dim[0], dim[1], v_phi, u_phi, v_mag, u_mag, phase, threshold)
    # Return the phase:
    return phase


def phase_mag_fourier(a, projection, padding=0, b_0=1):
    '''Calculate the magnetic phase from magnetization data (Fourier space approach).

    Parameters
    ----------
    a : float
        The grid spacing in nm.
    projection : tuple (N=3) of :class:`~numpy.ndarray` (N=2)
        The in-plane projection of the magnetization as a tuple, storing the `u`- and `v`-component
        of the magnetization and the thickness projection for the resulting 2D-grid.
    padding : int, optional
        Factor for the zero padding. The default is 0 (no padding). For a factor of n the number
        of pixels is increase by ``(1+n)**2``. Padded zeros are cropped at the end.
    b_0 : float, optional
        The magnetic induction corresponding to a magnetization `M`\ :sub:`0` in T.
        The default is 1.

    Returns
    -------
    phase : :class:`~numpy.ndarray` (N=2)
        The phase as a 2-dimensional array.

    '''
    v_dim, u_dim = np.shape(projection[0])
    v_mag, u_mag = projection[:-1]
    # Create zero padded matrices:
    u_pad = u_dim/2 * padding
    v_pad = v_dim/2 * padding
    u_mag_big = np.zeros(((1 + padding) * v_dim, (1 + padding) * u_dim))
    v_mag_big = np.zeros(((1 + padding) * v_dim, (1 + padding) * u_dim))
    u_mag_big[v_pad:v_pad+v_dim, u_pad:u_pad+u_dim] = u_mag
    v_mag_big[v_pad:v_pad+v_dim, u_pad:u_pad+u_dim] = v_mag
    # Fourier transform of the two components:
    u_mag_fft = np.fft.fftshift(np.fft.rfft2(u_mag_big), axes=0)
    v_mag_fft = np.fft.fftshift(np.fft.rfft2(v_mag_big), axes=0)
    # Calculate the Fourier transform of the phase:
    f_nyq = 0.5 / a  # nyquist frequency
    f_u = np.linspace(0, f_nyq, u_mag_fft.shape[1])
    f_v = np.linspace(-f_nyq, f_nyq, u_mag_fft.shape[0], endpoint=False)
    f_uu, f_vv = np.meshgrid(f_u, f_v)
    coeff = (1j*b_0) / (2*PHI_0)
    phase_fft = coeff * a * (u_mag_fft*f_vv - v_mag_fft*f_uu) / (f_uu**2 + f_vv**2 + 1e-30)
    # Transform to real space and revert padding:
    phase_big = np.fft.irfft2(np.fft.ifftshift(phase_fft, axes=0))
    phase = phase_big[v_pad:v_pad+v_dim, u_pad:u_pad+u_dim]
    return phase

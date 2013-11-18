# -*- coding: utf-8 -*-
"""Create magnetic and electric phase maps from magnetization data.

This module executes several forward models to calculate the magnetic or electric phase map from
a given projection of a 3-dimensional magnetic distribution (see :mod:`~pyramid.projector`).
For the magnetic phase map, an approach using real space and one using Fourier space is provided.
The real space approach also calculates the Jacobi matrix, which is used for the reconstruction in
the :mod:`~pyramid.reconstructor` module. The electrostatic contribution is calculated by using
the assumption of a mean inner potentail (MIP).

"""


import numpy as np
from numpy import pi

import pyramid.numcore as nc

from scipy.signal import fftconvolve


PHI_0 = -2067.83    # magnetic flux in T*nm²
H_BAR = 6.626E-34  # Planck constant in J*s
M_E = 9.109E-31    # electron mass in kg
Q_E = 1.602E-19    # electron charge in C
C = 2.998E8        # speed of light in m/s


def phase_mag_fourier(res, projection, padding=0, b_0=1):
    '''Calculate the magnetic phase from magnetization data (Fourier space approach).

    Parameters
    ----------
    res : float
        The resolution of the grid (grid spacing) in nm.
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
    nyq = 0.5 / res  # nyquist frequency
    f_u = np.linspace(0, nyq, u_mag_fft.shape[1])
    f_v = np.linspace(-nyq, nyq, u_mag_fft.shape[0], endpoint=False)
    f_uu, f_vv = np.meshgrid(f_u, f_v)
    coeff = (1j*b_0) / (2*PHI_0)
    phase_fft = coeff * res * (u_mag_fft*f_vv - v_mag_fft*f_uu) / (f_uu**2 + f_vv**2 + 1e-30) #* (8*(res/2)**3)*np.sinc(res/2*f_uu)*np.sinc(res/2*f_vv) * np.exp(2*pi*1j*())
    # Transform to real space and revert padding:
    phase_big = np.fft.irfft2(np.fft.ifftshift(phase_fft, axes=0))
    phase = phase_big[v_pad:v_pad+v_dim, u_pad:u_pad+u_dim]
    return phase


def phase_mag_real(res, projection, method='discs', b_0=1, jacobi=None):
    '''Calculate the magnetic phase from magnetization data (real space approach).

    Parameters
    ----------
    res : float
        The resolution of the grid (grid spacing) in nm.
    projection : tuple (N=3) of :class:`~numpy.ndarray` (N=2)
        The in-plane projection of the magnetization as a tuple, storing the `u`- and `v`-component
        of the magnetization and the thickness projection for the resulting 2D-grid.
    method : {'disc', 'slab'}, optional
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

    # Get lookup-tables for the phase of one pixel:
    u_phi = get_kernel(method, 'u', dim, res, b_0)
    v_phi = get_kernel(method, 'v', dim, res, b_0)

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


def phase_mag_real_conv(res, projection, method='disc', b_0=1):
    '''Calculate the magnetic phase from magnetization data (real space approach).

    Parameters
    ----------
    res : float
        The resolution of the grid (grid spacing) in nm.
    projection : tuple (N=3) of :class:`~numpy.ndarray` (N=2)
        The in-plane projection of the magnetization as a tuple, storing the `u`- and `v`-component
        of the magnetization and the thickness projection for the resulting 2D-grid.
    method : {'disc', 'slab'}, optional
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

    '''  # TODO: Docstring!!!
    # Process input parameters:
    dim = np.shape(projection[0])
    v_mag, u_mag = projection[:-1]

    # Get lookup-tables for the phase of one pixel:
    u_phi = get_kernel(method, 'u', dim, res, b_0)
    v_phi = get_kernel(method, 'v', dim, res, b_0)

    # Return the phase:
    result = fftconvolve(u_mag, u_phi, 'same') - fftconvolve(v_mag, v_phi, 'same')
    return result


def phase_mag_real_fast(res, projection, kernels_fourier, b_0=1):
    '''Calculate the magnetic phase from magnetization data (real space approach).

    Parameters
    ----------
    res : float
        The resolution of the grid (grid spacing) in nm.
    projection : tuple (N=3) of :class:`~numpy.ndarray` (N=2)
        The in-plane projection of the magnetization as a tuple, storing the `u`- and `v`-component
        of the magnetization and the thickness projection for the resulting 2D-grid.
    method : {'disc', 'slab'}, optional
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

    '''  # TODO: Docstring!!!
    # Process input parameters:
    v_mag, u_mag = projection[:-1]
    dim = np.shape(u_mag)

    size = 3*np.array(dim) - 1  # dim + (2*dim - 1) magnetisation + kernel
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    fslice = tuple([slice(0, int(sz)) for sz in size])

    u_mag_f = np.fft.rfftn(u_mag, fsize)
    v_mag_f = np.fft.rfftn(v_mag, fsize)

    v_kern_f, u_kern_f = kernels_fourier

    fslice = (slice(dim[0]-1, 2*dim[0]-1), slice(dim[1]-1, 2*dim[1]-1))
    u_phase = np.fft.irfftn(u_mag_f * u_kern_f, fsize)[fslice].copy()
    v_phase = np.fft.irfftn(v_mag_f * v_kern_f, fsize)[fslice].copy()
    return u_phase - v_phase
    
    
def get_kernel_fourier(method, orientation, dim, res, b_0=1):

    kernel = get_kernel(method, orientation, dim, res, b_0)
 
    size = 3*np.array(dim) - 1  # dim + (2*dim - 1) magnetisation + kernel
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)

    return np.fft.rfftn(kernel, fsize)


def get_kernel(method, orientation, dim, res, b_0=1):
    
    def get_elementary_phase(method, n, m, res):
        if method == 'slab':
            def F_h(n, m):
                a = np.log(res**2 * (n**2 + m**2))
                b = np.arctan(n / m)
                return n*a - 2*n + 2*m*b
            return 0.5 * (F_h(n-0.5, m-0.5) - F_h(n+0.5, m-0.5)
                        - F_h(n-0.5, m+0.5) + F_h(n+0.5, m+0.5))
        elif method == 'disc':
            in_or_out = np.logical_not(np.logical_and(n == 0, m == 0))
            return m / (n**2 + m**2 + 1E-30) * in_or_out
    
    coeff = -b_0 * res**2 / (2*PHI_0)
    v_dim, u_dim = dim
    u = np.linspace(-(u_dim-1), u_dim-1, num=2*u_dim-1)
    v = np.linspace(-(v_dim-1), v_dim-1, num=2*v_dim-1)
    uu, vv = np.meshgrid(u, v)
    if orientation == 'u':
        return coeff * get_elementary_phase(method, uu, vv, res)
    elif orientation == 'v':
        return coeff * get_elementary_phase(method, vv, uu, res)


def phase_elec(res, projection, v_0=1, v_acc=30000):
    '''Calculate the electric phase from magnetization distributions.

    Parameters
    ----------
    res : float
        The resolution of the grid (grid spacing) in nm.
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
    return v_0 * Ce * projection[-1] * res*1E-9

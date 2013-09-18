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
    nyq = 1 / res  # nyquist frequency
    f_u = np.linspace(0, nyq/2, u_mag_fft.shape[1])
    f_v = np.linspace(-nyq/2, nyq/2, u_mag_fft.shape[0], endpoint=False)
    f_uu, f_vv = np.meshgrid(f_u, f_v)
    coeff = (1j*b_0) / (2*PHI_0)
    phase_fft = coeff * res * (u_mag_fft*f_vv - v_mag_fft*f_uu) / (f_uu**2 + f_vv**2 + 1e-30)
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

    # Function for creating the lookup-tables:
    def phi_lookup(method, n, m, res, b_0):
        if method == 'slab':
            def F_h(n, m):
                a = np.log(res**2 * (n**2 + m**2))
                b = np.arctan(n / m)
                return n*a - 2*n + 2*m*b
            return coeff * 0.5 * (F_h(n-0.5, m-0.5) - F_h(n+0.5, m-0.5)
                                - F_h(n-0.5, m+0.5) + F_h(n+0.5, m+0.5))
        elif method == 'disc':
            in_or_out = np.logical_not(np.logical_and(n == 0, m == 0))
            return coeff * m / (n**2 + m**2 + 1E-30) * in_or_out

    # Process input parameters:
    v_dim, u_dim = np.shape(projection[0])
    v_mag, u_mag = projection[:-1]
    coeff = -b_0 * res**2 / (2*PHI_0)

    # Create lookup-tables for the phase of one pixel:
    u = np.linspace(-(u_dim-1), u_dim-1, num=2*u_dim-1)
    v = np.linspace(-(v_dim-1), v_dim-1, num=2*v_dim-1)
    uu, vv = np.meshgrid(u, v)
    u_phi = phi_lookup(method, uu, vv, res, b_0)
    v_phi = phi_lookup(method, vv, uu, res, b_0)

    # Calculation of the phase:
    phase = np.zeros((v_dim, u_dim))
    threshold = 0
    if jacobi is not None:  # With Jacobian matrix (slower)
        jacobi[:] = 0  # Jacobi matrix --> zeros
        for j in range(v_dim):
            for i in range(u_dim):
                u_phase = u_phi[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]
                jacobi[:, i+u_dim*j] = u_phase.reshape(-1)
                if abs(u_mag[j, i]) > threshold:
                    phase += u_mag[j, i] * u_phase
                v_phase = v_phi[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]
                jacobi[:, u_dim*v_dim+i+u_dim*j] = -v_phase.reshape(-1)
                if abs(v_mag[j, i]) > threshold:
                    phase -= v_mag[j, i] * v_phase
    else:  # Without Jacobi matrix (faster)
        nc.phase_mag_real_core(v_dim, u_dim, v_phi, u_phi, v_mag, u_mag, phase, threshold)
    # Return the phase:
    return phase


def phase_mag_real_conv(res, projection, method='disc', b_0=1, jacobi=None):
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

    # Function for creating the lookup-tables:
    def phi_lookup(method, n, m, res, b_0):
        if method == 'slab':
            def F_h(n, m):
                a = np.log(res**2 * (n**2 + m**2))
                b = np.arctan(n / m)
                return n*a - 2*n + 2*m*b
            return coeff * 0.5 * (F_h(n-0.5, m-0.5) - F_h(n+0.5, m-0.5)
                                - F_h(n-0.5, m+0.5) + F_h(n+0.5, m+0.5))
        elif method == 'disc':
            in_or_out = np.logical_not(np.logical_and(n == 0, m == 0))
            return coeff * m / (n**2 + m**2 + 1E-30) * in_or_out

    # Process input parameters:
    v_dim, u_dim = np.shape(projection[0])
    v_mag, u_mag = projection[:-1]
    coeff = -b_0 * res**2 / (2*PHI_0)

    # Create lookup-tables for the phase of one pixel:
    u = np.linspace(-(u_dim-1), u_dim-1, num=2*u_dim-1)
    v = np.linspace(-(v_dim-1), v_dim-1, num=2*v_dim-1)
    uu, vv = np.meshgrid(u, v)
    u_phi = phi_lookup(method, uu, vv, res, b_0)
    v_phi = phi_lookup(method, vv, uu, res, b_0)

    # Return the phase:
    return fftconvolve(u_mag, u_phi, 'same') - fftconvolve(v_mag, v_phi, 'same')

#    def fftconvolve(in1, in2, mode="full"):
#    """Convolve two N-dimensional arrays using FFT.
#
#    Convolve `in1` and `in2` using the fast Fourier transform method, with
#    the output size determined by the `mode` argument.
#
#    This is generally much faster than `convolve` for large arrays (n > ~500),
#    but can be slower when only a few output values are needed, and can only
#    output float arrays (int or object array inputs will be cast to float).
#
#    Parameters
#    ----------
#    in1 : array_like
#        First input.
#    in2 : array_like
#        Second input. Should have the same number of dimensions as `in1`;
#        if sizes of `in1` and `in2` are not equal then `in1` has to be the
#        larger array.
#    mode : str {'full', 'valid', 'same'}, optional
#        A string indicating the size of the output:
#
#        ``full``
#           The output is the full discrete linear convolution
#           of the inputs. (Default)
#        ``valid``
#           The output consists only of those elements that do not
#           rely on the zero-padding.
#        ``same``
#           The output is the same size as `in1`, centered
#           with respect to the 'full' output.
#
#    Returns
#    -------
#    out : array
#        An N-dimensional array containing a subset of the discrete linear
#        convolution of `in1` with `in2`.
#
#    """
#    in1 = asarray(in1)
#    in2 = asarray(in2)
#
#    if rank(in1) == rank(in2) == 0:  # scalar inputs
#        return in1 * in2
#    elif not in1.ndim == in2.ndim:
#        raise ValueError("in1 and in2 should have the same rank")
#    elif in1.size == 0 or in2.size == 0:  # empty arrays
#        return array([])
#
#    s1 = array(in1.shape)
#    s2 = array(in2.shape)
#    complex_result = (np.issubdtype(in1.dtype, np.complex) or
#                      np.issubdtype(in2.dtype, np.complex))
#    size = s1 + s2 - 1
#
#    if mode == "valid":
#        for d1, d2 in zip(s1, s2):
#            if not d1 >= d2:
#                warnings.warn("in1 should have at least as many items as in2 in "
#                              "every dimension for 'valid' mode.  In scipy "
#                              "0.13.0 this will raise an error",
#                              DeprecationWarning)
#
#    # Always use 2**n-sized FFT
#    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
#    print('fsize =', fsize)
#    print('s1 =', s1)
#    print('s2 =', s2)
#    fslice = tuple([slice(0, int(sz)) for sz in size])
#    if not complex_result:
#        ret = irfftn(rfftn(in1, fsize) *
#                     rfftn(in2, fsize), fsize)[fslice].copy()
#        ret = ret.real
#    else:
#        ret = ifftn(fftn(in1, fsize) * fftn(in2, fsize))[fslice].copy()
#
#    if mode == "full":
#        return ret
#    elif mode == "same":
#        return _centered(ret, s1)
#    elif mode == "valid":
#        return _centered(ret, abs(s1 - s2) + 1)


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


#def phase_mag_real2(res, projection, method, b_0=1, jacobi=None):
#    '''Calculate phasemap from magnetization data (real space approach).
#    Arguments:
#        res        - the resolution of the grid (grid spacing) in nm
#        projection - projection of a magnetic distribution (created with pyramid.projector)
#        method     - string, describing the method to use for the pixel field ('slab' or 'disc')
#        b_0        - magnetic induction corresponding to a magnetization Mo in T (default: 1)
#        jacobi     - matrix in which to save the jacobi matrix (default: None, faster computation)
#    Returns:
#        the phasemap as a 2 dimensional array
#
#    '''
#    # Function for creating the lookup-tables:
#    def phi_lookup(method, n, m, res, b_0):
#        if method == 'slab':
#            def F_h(n, m):
#                a = np.log(res**2 * (n**2 + m**2))
#                b = np.arctan(n / m)
#                return n*a - 2*n + 2*m*b
#            return coeff * 0.5 * (F_h(n-0.5, m-0.5) - F_h(n+0.5, m-0.5)
#                                - F_h(n-0.5, m+0.5) + F_h(n+0.5, m+0.5))
#        elif method == 'disc':
#            in_or_out = np.logical_not(np.logical_and(n == 0, m == 0))
#            return coeff * m / (n**2 + m**2 + 1E-30) * in_or_out
#    # Process input parameters:
#    v_dim, u_dim = np.shape(projection[0])
#    v_mag, u_mag = projection
#    coeff = -b_0 * res**2 / (2*PHI_0)
#    # Create lookup-tables for the phase of one pixel:
#    u = np.linspace(-(u_dim-1), u_dim-1, num=2*u_dim-1)
#    v = np.linspace(-(v_dim-1), v_dim-1, num=2*v_dim-1)
#    uu, vv = np.meshgrid(u, v)
#    phi_u = phi_lookup(method, uu, vv, res, b_0)
#    phi_v = phi_lookup(method, vv, uu, res, b_0)
#    # Calculation of the phase:
#    phase = np.zeros((v_dim, u_dim))
#    threshold = 0
#    if jacobi is not None:  # With Jacobian matrix (slower)
#        jacobi[:] = 0  # Jacobi matrix --> zeros
#        ############################### TODO: NUMERICAL CORE  ####################################
#        for j in range(v_dim):
#            for i in range(u_dim):
#                phase_u = phi_u[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]
#                jacobi[:, i+u_dim*j] = phase_u.reshape(-1)
#                if abs(u_mag[j, i]) > threshold:
#                    phase += u_mag[j, i] * phase_u
#                phase_v = phi_v[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]
#                jacobi[:, u_dim*v_dim+i+u_dim*j] = -phase_v.reshape(-1)
#                if abs(v_mag[j, i]) > threshold:
#                    phase -= v_mag[j, i] * phase_v
#        ############################### TODO: NUMERICAL CORE  ####################################
#    else:  # Without Jacobi matrix (faster)
##        phasecopy = phase.copy()
##        start_time = time.time()
##        numcore.phase_mag_real_helper_1(v_dim, u_dim, phi_u, phi_v,
##                                        u_mag, v_mag, phasecopy, threshold)
##        print 'with numcore   : ', time.time() - start_time
##        start_time = time.time()
#        for j in range(v_dim):
#            for i in range(u_dim):
#                if abs(u_mag[j, i]) > threshold:
#                    phase += u_mag[j, i] * phi_u[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]
#                if abs(v_mag[j, i]) > threshold:
#                    phase -= v_mag[j, i] * phi_v[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]
##        print 'without numcore: ', time.time() - start_time
##        print 'num. difference: ', ((phase - phasecopy) ** 2).sum()
#    # Return the phase:
#    return phase
#
#
#def phase_mag_real_alt(res, projection, method, b_0=1, jacobi=None):  # TODO: Modify
#    '''Calculate phasemap from magnetization data (real space approach).
#    Arguments:
#        res        - the resolution of the grid (grid spacing) in nm
#        projection - projection of a magnetic distribution (created with pyramid.projector)
#        method     - string, describing the method to use for the pixel field ('slab' or 'disc')
#        b_0        - magnetic induction corresponding to a magnetization Mo in T (default: 1)
#        jacobi     - matrix in which to save the jacobi matrix (default: None, faster computation)
#    Returns:
#        the phasemap as a 2 dimensional array
#
#    '''
#    # Function for creating the lookup-tables:
#    def phi_lookup(method, n, m, res, b_0):
#        if method == 'slab':
#            def F_h(n, m):
#                a = np.log(res**2 * (n**2 + m**2))
#                b = np.arctan(n / m)
#                return n*a - 2*n + 2*m*b
#            return coeff * 0.5 * (F_h(n-0.5, m-0.5) - F_h(n+0.5, m-0.5)
#                                - F_h(n-0.5, m+0.5) + F_h(n+0.5, m+0.5))
#        elif method == 'disc':
#            in_or_out = np.logical_not(np.logical_and(n == 0, m == 0))
#            return coeff * m / (n**2 + m**2 + 1E-30) * in_or_out
#
#    # Function for the phase contribution of one pixel:
#    def phi_mag(i, j):
#        return (np.cos(beta[j, i]) * phi_u[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]
#              - np.sin(beta[j, i]) * phi_v[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i])
#
#    # Function for the derivative of the phase contribution of one pixel:
#    def phi_mag_deriv(i, j):
#        return -(np.sin(beta[j, i]) * phi_u[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i]
#               + np.cos(beta[j, i]) * phi_v[v_dim-1-j:(2*v_dim-1)-j, u_dim-1-i:(2*u_dim-1)-i])
#
#    # Process input parameters:
#    v_dim, u_dim = np.shape(projection[0])
#    v_mag, u_mag = projection
#    beta = np.arctan2(v_mag, u_mag)
#    mag = np.hypot(u_mag, v_mag)
#    coeff = -b_0 * res**2 / (2*PHI_0)
#    # Create lookup-tables for the phase of one pixel:
#    u = np.linspace(-(u_dim-1), u_dim-1, num=2*u_dim-1)
#    v = np.linspace(-(v_dim-1), v_dim-1, num=2*v_dim-1)
#    uu, vv = np.meshgrid(u, v)
#    phi_u = phi_lookup(method, uu, vv, res, b_0)
#    phi_v = phi_lookup(method, vv, uu, res, b_0)
#    # Calculation of the phase:
#    phase = np.zeros((v_dim, u_dim))
#    threshold = 0
#    if jacobi is not None:  # With Jacobian matrix (slower)
#        jacobi[:] = 0  # Jacobi matrix --> zeros
#        ############################### TODO: NUMERICAL CORE  ####################################
#        for j in range(v_dim):
#            for i in range(u_dim):
#                phase_cache = phi_mag(i, j)
#                jacobi[:, i+u_dim*j] = phase_cache.reshape(-1)
#                if mag[j, i] > threshold:
#                    phase += mag[j, i]*phase_cache
#                    jacobi[:, u_dim*v_dim+i+u_dim*j] = (mag[j, i]*phi_mag_deriv(i, j)).reshape(-1)
#        ############################### TODO: NUMERICAL CORE  ####################################
#    else:  # Without Jacobi matrix (faster)
#        for j in range(v_dim):
#            for i in range(u_dim):
#                if abs(mag[j, i]) > threshold:
#                    phase += mag[j, i] * phi_mag(i, j)
#    # Return the phase:
#    return phase

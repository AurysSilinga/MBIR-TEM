# -*- coding: utf-8 -*-
"""Create magnetic and electric phase maps from magnetization data.

This module executes several forward models to calculate the magnetic or electric phase map from
a given projection of a 3-dimensional magnetic distribution (see :mod:`~pyramid.projector`).
For the magnetic phase map, an approach using real space and one using Fourier space is provided.
The electrostatic contribution is calculated by using the assumption of a mean inner potential.

"""


import numpy as np
from numpy import pi

import abc

import pyramid.numcore.kernel_core as nc
from pyramid.kernel import Kernel
from pyramid.projector import Projector
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap
from pyramid.forwardmodel import ForwardModel





class PhaseMapper(object):

    '''DOCSTRING'''  # TODO: Docstring!

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, projector):
        '''Docstring: has to be implemented!'''  # TODO: Docstring!
    
    @abc.abstractmethod
    def __call__(self, mag_data):
        '''Docstring: has to be implemented!'''  # TODO: Docstring!


class PMAdapterFM(PhaseMapper):

    def __init__(self, a, projector, b_0=1, geometry='disc'):
        assert isinstance(projector, Projector), 'Argument has to be a Projector object!'
        self.a = a
        self.projector = projector
        self.fwd_model = ForwardModel([projector], Kernel(a, projector.dim_2d, b_0, geometry))

    def __call__(self, mag_data):
        assert isinstance(mag_data, MagData), 'Only MagData objects can be mapped!'
        assert  mag_data.a == self.a, 'Grid spacing has to match!'
        # TODO: test if mag_data fits in all aspects
        phase_map = PhaseMap(self.a, np.zeros(self.projector.dim_2d))
        phase_map.phase_vec = self.fwd_model(mag_data.mag_vec)
        return phase_map


class PMFourier(PhaseMapper):
    
    PHI_0 = -2067.83   # magnetic flux in T*nm²

    def __init__(self, a, projector, b_0=1, padding=0):
        assert isinstance(projector, Projector), 'Argument has to be a Projector object!'
        self.a = a
        self.projector = projector
        self.b_0 = b_0
        self.padding = padding        

    def __call__(self, mag_data):
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
    
        '''  # TODO: Docstring
        assert isinstance(mag_data, MagData), 'Only MagData objects can be mapped!'
        assert  mag_data.a == self.a, 'Grid spacing has to match!'
        # TODO: test if mag_data fits in all aspects (especially with projector)
        v_dim, u_dim = self.projector.dim_2d
        u_mag, v_mag = self.projector(mag_data.mag_vec).reshape((2,)+self.projector.dim_2d)
        # Create zero padded matrices:
        u_pad = u_dim/2 * self.padding
        v_pad = v_dim/2 * self.padding
        # TODO: use mag_data.padding or np.pad(...)
        u_mag_big = np.zeros(((1 + self.padding) * v_dim, (1 + self.padding) * u_dim))
        v_mag_big = np.zeros(((1 + self.padding) * v_dim, (1 + self.padding) * u_dim))
        u_mag_big[v_pad:v_pad+v_dim, u_pad:u_pad+u_dim] = u_mag
        v_mag_big[v_pad:v_pad+v_dim, u_pad:u_pad+u_dim] = v_mag
        # Fourier transform of the two components:
        u_mag_fft = np.fft.fftshift(np.fft.rfft2(u_mag_big), axes=0)
        v_mag_fft = np.fft.fftshift(np.fft.rfft2(v_mag_big), axes=0)
        # Calculate the Fourier transform of the phase:
        f_nyq = 0.5 / self.a  # nyquist frequency
        f_u = np.linspace(0, f_nyq, u_mag_fft.shape[1])
        f_v = np.linspace(-f_nyq, f_nyq, u_mag_fft.shape[0], endpoint=False)
        f_uu, f_vv = np.meshgrid(f_u, f_v)
        coeff = (1j*self.b_0) / (2*self.PHI_0)
        phase_fft = coeff*self.a * (u_mag_fft*f_vv - v_mag_fft*f_uu) / (f_uu**2 + f_vv**2 + 1e-30)
        # Transform to real space and revert padding:
        phase_big = np.fft.irfft2(np.fft.ifftshift(phase_fft, axes=0))
        phase = phase_big[v_pad:v_pad+v_dim, u_pad:u_pad+u_dim]
        return PhaseMap(self.a, phase)


class PMElectric(PhaseMapper):
    
    H_BAR = 6.626E-34  # Planck constant in J*s
    M_E = 9.109E-31    # electron mass in kg
    Q_E = 1.602E-19    # electron charge in C
    C = 2.998E8        # speed of light in m/s
    
    def __init__(self, a, projector, v_0=1, v_acc=30000, threshold=0):
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
        
        '''  # Docstring!
        self.a = a
        self.projector = projector
        self.v_0 = v_0
        self.v_acc = v_acc
        self.threshold = threshold

    def __call__(self, mag_data):
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

        '''  # Docstring!
        # Coefficient calculation:
        H_BAR, M_E, Q_E, C = self.H_BAR, self.M_E, self.Q_E, self.C
        v_0, v_acc = self.v_0, self.v_acc
        lam = H_BAR / np.sqrt(2 * M_E * Q_E * v_acc * (1 + Q_E*v_acc / (2*M_E*C**2)))
        Ce = 2*pi*Q_E/lam * (Q_E*v_acc + M_E*C**2) / (Q_E*v_acc * (Q_E*v_acc + 2*M_E*C**2))
        # Calculate mask:
        mask = np.sqrt(np.sum(np.array(mag_data.magnitude)**2, axis=0)) > self.threshold
        # Project and calculate phase:
        projection = self.projector(mask.reshape(-1)).reshape(self.projector.dim)
        phase = v_0 * Ce * projection * self.a*1E-9
        return PhaseMap(self.a, phase)


class PMConvolve(PhaseMapper):

    # TODO: Docstring!

    def __init__(self, a, projector, b_0=1, threshold=0, geometry='disc'):
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

        '''# Docstring!
        assert isinstance(projector, Projector), 'Argument has to be a Projector object!'
        self.a = a
        self.projector = projector
        self.threshold = threshold
        self.kernel = Kernel(a, projector.dim_2d, b_0, geometry)

    def __call__(self, mag_data):
        # Docstring!
        # Process input parameters:
        u_mag, v_mag = self.projector(mag_data.mag_vec).reshape((2,)+self.projector.dim_2d)    
        # Fourier transform the projected magnetisation:
        kernel = self.kernel
        u_mag_fft = np.fft.rfftn(u_mag, kernel.dim_fft)
        v_mag_fft = np.fft.rfftn(v_mag, kernel.dim_fft)
        # Convolve the magnetization with the kernel in Fourier space:
        u_phase = np.fft.irfftn(u_mag_fft * kernel.u_fft, kernel.dim_fft)[kernel.slice_fft].copy()
        v_phase = np.fft.irfftn(v_mag_fft * kernel.v_fft, kernel.dim_fft)[kernel.slice_fft].copy()
        # Return the result:
        return PhaseMap(self.a, u_phase - v_phase)


class PMReal(PhaseMapper):

    def __init__(self, a, projector, b_0=1, threshold=0, geometry='disc', numcore=True):
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
    
        '''  # Docstring!
        assert isinstance(projector, Projector), 'Argument has to be a Projector object!'
        self.a = a
        self.projector = projector
        self.threshold = threshold
        self.kernel = Kernel(a, projector.dim_2d, b_0, geometry)
        self.numcore = numcore

    def __call__(self, mag_data):
        # TODO: Docstring
        # Process input parameters: 
        dim = self.projector.dim_2d
        threshold = self.threshold
        u_mag, v_mag = self.projector(mag_data.mag_vec).reshape((2,)+dim)
        # Create kernel (lookup-tables for the phase of one pixel):
        u_phi = self.kernel.u
        v_phi = self.kernel.v
        # Calculation of the phase:
        phase = np.zeros(dim)
        if self.numcore:
            nc.phase_mag_real_core(dim[0], dim[1], v_phi, u_phi, v_mag, u_mag, phase, threshold)
        else:
            for j in range(dim[0]):
                for i in range(dim[1]):
                    u_phase = u_phi[dim[0]-1-j:(2*dim[0]-1)-j, dim[1]-1-i:(2*dim[1]-1)-i]
                    if abs(u_mag[j, i]) > threshold:
                        phase += u_mag[j, i] * u_phase
                    v_phase = v_phi[dim[0]-1-j:(2*dim[0]-1)-j, dim[1]-1-i:(2*dim[1]-1)-i]
                    if abs(v_mag[j, i]) > threshold:
                        phase -= v_mag[j, i] * v_phase
        # Return the phase:
        return PhaseMap(self.a, phase)

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
from pyramid.kernel import Kernel
from pyramid.projector import Projector
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap
from pyramid.forwardmodel import ForwardModel

import logging


class PhaseMapper(object):

    '''Abstract base class for the phase calculation from a 2-dimensional distribution.

    The :class:`~.PhaseeMapper-` class represents a strategy for the phasemapping of a
    2-dimensional magnetic distribution with two components onto a scalar phase map.
    :class:`~.Kernel` is an abstract base class and provides a unified interface which should be
    subclassed with custom :func:`__init__` and :func:`__call__` functions. Concrete subclasses
    can be called as a function and take a :class:`~.MagData` object as input and return a
    :class:`~.PhaseMap` object.

    '''

    __metaclass__ = abc.ABCMeta
    LOG = logging.getLogger(__name__+'.PhaseMapper')

    @abc.abstractmethod
    def __init__(self, projector):
        self.LOG.debug('Calling __init__')
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self, mag_data):
        self.LOG.debug('Calling __call__')
        raise NotImplementedError()


class PMAdapterFM(PhaseMapper):

    '''Class representing a phase mapping strategy adapting the :class:`~.ForwardModel` class.

    The :class:`~.PMAdapterFM` class is an adapter class to incorporate the forward model from the
    :mod:`~.ForwardModel` module without the need of vector input and output. It directly takes
    :class:`~.MagData` objects and returns :class:`~.PhaseMap` objects.

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
    fwd_model : :class:`~.ForwardModel`
        The forward model that is constructed from the given information. It is created internally
        and does not have to be provided by the user.

    '''

    LOG = logging.getLogger(__name__+'.PMAdapterFM')

    def __init__(self, a, projector, b_0=1, geometry='disc'):
        self.LOG.debug('Calling __init__')
        assert isinstance(projector, Projector), 'Argument has to be a Projector object!'
        self.a = a
        self.projector = projector
        self.b_0 = b_0
        self.geometry = geometry
        self.fwd_model = ForwardModel([projector], Kernel(a, projector.dim_uv, b_0, geometry))
        self.LOG.debug('Created '+str(self))

    def __call__(self, mag_data):
        self.LOG.debug('Calling __call__')
        assert isinstance(mag_data, MagData), 'Only MagData objects can be mapped!'
        assert mag_data.a == self.a, 'Grid spacing has to match!'
        assert mag_data.dim == self.projector.dim, 'Dimensions do not match!'
        phase_map = PhaseMap(self.a, np.zeros(self.projector.dim_uv))
        phase_map.phase_vec = self.fwd_model(mag_data.mag_vec)
        return phase_map

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(a=%r, projector=%r, b_0=%r, geometry=%r)' % \
            (self.__class__, self.a, self.projector, self.b_0, self.geometry)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'PMAdapterFM(a=%s, projector=%s, b_0=%s, geometry=%s)' % \
            (self.a, self.projector, self.b_0, self.geometry)


class PMFourier(PhaseMapper):

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

    LOG = logging.getLogger(__name__+'.PMFourier')
    PHI_0 = -2067.83   # magnetic flux in T*nm²

    def __init__(self, a, projector, b_0=1, padding=0):
        self.LOG.debug('Calling __init__')
        assert isinstance(projector, Projector), 'Argument has to be a Projector object!'
        self.a = a
        self.projector = projector
        self.b_0 = b_0
        self.padding = padding
        self.LOG.debug('Created '+str(self))

    def __call__(self, mag_data):
        self.LOG.debug('Calling __call__')
        assert isinstance(mag_data, MagData), 'Only MagData objects can be mapped!'
        assert mag_data.a == self.a, 'Grid spacing has to match!'
        assert mag_data.dim == self.projector.dim, 'Dimensions do not match!'
        v_dim, u_dim = self.projector.dim_uv
        u_mag, v_mag = self.projector(mag_data.mag_vec).reshape((2,)+self.projector.dim_uv)
        # Create zero padded matrices:
        u_pad = int(u_dim/2 * self.padding)
        v_pad = int(v_dim/2 * self.padding)
        u_mag_pad = np.pad(u_mag, ((v_pad, v_pad), (u_pad, u_pad)), 'constant', constant_values=0)
        v_mag_pad = np.pad(v_mag, ((v_pad, v_pad), (u_pad, u_pad)), 'constant', constant_values=0)
        # Fourier transform of the two components:
        u_mag_fft = np.fft.fftshift(np.fft.rfft2(u_mag_pad), axes=0)
        v_mag_fft = np.fft.fftshift(np.fft.rfft2(v_mag_pad), axes=0)
        # Calculate the Fourier transform of the phase:
        f_nyq = 0.5 / self.a  # nyquist frequency
        f_u = np.linspace(0, f_nyq, u_mag_fft.shape[1])
        f_v = np.linspace(-f_nyq, f_nyq, u_mag_fft.shape[0], endpoint=False)
        f_uu, f_vv = np.meshgrid(f_u, f_v)
        coeff = (1j*self.b_0) / (2*self.PHI_0)
        phase_fft = coeff*self.a * (u_mag_fft*f_vv - v_mag_fft*f_uu) / (f_uu**2 + f_vv**2 + 1e-30)
        # Transform to real space and revert padding:
        phase_pad = np.fft.irfft2(np.fft.ifftshift(phase_fft, axes=0))
        phase = phase_pad[v_pad:v_pad+v_dim, u_pad:u_pad+u_dim]
        return PhaseMap(self.a, phase)

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(a=%r, projector=%r, b_0=%r, padding=%r)' % \
            (self.__class__, self.a, self.projector, self.b_0, self.padding)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'PMFourier(a=%s, projector=%s, b_0=%s, padding=%s)' % \
            (self.a, self.projector, self.b_0, self.padding)

class PMElectric(PhaseMapper):

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

    LOG = logging.getLogger(__name__+'.PMElectric')
    H_BAR = 6.626E-34  # Planck constant in J*s
    M_E = 9.109E-31    # electron mass in kg
    Q_E = 1.602E-19    # electron charge in C
    C = 2.998E8        # speed of light in m/s

    def __init__(self, a, projector, v_0=1, v_acc=30000, threshold=0):
        self.LOG.debug('Calling __init__')
        assert isinstance(projector, Projector), 'Argument has to be a Projector object!'
        self.a = a
        self.projector = projector
        self.v_0 = v_0
        self.v_acc = v_acc
        self.threshold = threshold
        self.LOG.debug('Created '+str(self))

    def __call__(self, mag_data):
        self.LOG.debug('Calling __call__')
        assert isinstance(mag_data, MagData), 'Only MagData objects can be mapped!'
        assert mag_data.a == self.a, 'Grid spacing has to match!'
        assert mag_data.dim == self.projector.dim, 'Dimensions do not match!'
        # Coefficient calculation:
        H_BAR, M_E, Q_E, C = self.H_BAR, self.M_E, self.Q_E, self.C
        v_0, v_acc = self.v_0, self.v_acc
        lam = H_BAR / np.sqrt(2 * M_E * Q_E * v_acc * (1 + Q_E*v_acc / (2*M_E*C**2)))
        Ce = 2*pi*Q_E/lam * (Q_E*v_acc + M_E*C**2) / (Q_E*v_acc * (Q_E*v_acc + 2*M_E*C**2))
        # Calculate mask:
        mask = np.sqrt(np.sum(np.array(mag_data.magnitude)**2, axis=0)) > self.threshold
        # Project and calculate phase:
        projection = self.projector(mask.reshape(-1)).reshape(self.projector.dim_uv)
        phase = v_0 * Ce * projection * self.a*1E-9
        return PhaseMap(self.a, phase)

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(a=%r, projector=%r, v_0=%r, v_acc=%r, threshold=%r)' % \
            (self.__class__, self.a, self.projector, self.v_0, self.v_acc, self.threshold)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'PMElectric(a=%s, projector=%s, v_0=%s, v_acc=%s, threshold=%s)' % \
            (self.a, self.projector, self.v_0, self.v_acc, self.threshold)

class PMConvolve(PhaseMapper):

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

    '''

    LOG = logging.getLogger(__name__+'.PMConvolve')

    def __init__(self, a, projector, b_0=1, geometry='disc'):
        self.LOG.debug('Calling __init__')
        assert isinstance(projector, Projector), 'Argument has to be a Projector object!'
        self.a = a
        self.projector = projector
        self.b_0 = b_0
        self.geometry = geometry
        self.kernel = Kernel(a, projector.dim_uv, b_0, geometry)
        self.LOG.debug('Created '+str(self))

    def __call__(self, mag_data):
        self.LOG.debug('Calling __call__')
        assert isinstance(mag_data, MagData), 'Only MagData objects can be mapped!'
        assert mag_data.a == self.a, 'Grid spacing has to match!'
        assert mag_data.dim == self.projector.dim, 'Dimensions do not match!'
        # Process input parameters:
        u_mag, v_mag = self.projector(mag_data.mag_vec).reshape((2,)+self.projector.dim_uv)
        # Fourier transform the projected magnetisation:
        kernel = self.kernel
        u_mag_fft = np.fft.rfftn(u_mag, kernel.dim_fft)
        v_mag_fft = np.fft.rfftn(v_mag, kernel.dim_fft)
        # Convolve the magnetization with the kernel in Fourier space:
        u_phase = np.fft.irfftn(u_mag_fft * kernel.u_fft, kernel.dim_fft)[kernel.slice_fft].copy()
        v_phase = np.fft.irfftn(v_mag_fft * kernel.v_fft, kernel.dim_fft)[kernel.slice_fft].copy()
        # Return the result:
        return PhaseMap(self.a, u_phase-v_phase)

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(a=%r, projector=%r, b_0=%r, geometry=%r)' % \
            (self.__class__, self.a, self.projector, self.b_0, self.geometry)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'PMConvolve(a=%s, projector=%s, b_0=%s, geometry=%s)' % \
            (self.a, self.projector, self.b_0, self.geometry)


class PMReal(PhaseMapper):

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

    LOG = logging.getLogger(__name__+'.PMReal')

    def __init__(self, a, projector, b_0=1, threshold=0, geometry='disc', numcore=True):
        self.LOG.debug('Calling __init__')
        assert isinstance(projector, Projector), 'Argument has to be a Projector object!'
        self.a = a
        self.projector = projector
        self.b_0 = b_0
        self.threshold = threshold
        self.geometry = geometry
        self.kernel = Kernel(a, projector.dim_uv, b_0, geometry)
        self.numcore = numcore
        self.LOG.debug('Created '+str(self))

    def __call__(self, mag_data):
        self.LOG.debug('Calling __call__')
        assert isinstance(mag_data, MagData), 'Only MagData objects can be mapped!'
        assert mag_data.a == self.a, 'Grid spacing has to match!'
        assert mag_data.dim == self.projector.dim, 'Dimensions do not match!'
        # Process input parameters:
        dim_uv = self.projector.dim_uv
        threshold = self.threshold
        u_mag, v_mag = self.projector(mag_data.mag_vec).reshape((2,)+dim_uv)
        # Create kernel (lookup-tables for the phase of one pixel):
        u_phi = self.kernel.u
        v_phi = self.kernel.v
        # Calculation of the phase:
        phase = np.zeros(dim_uv)
        if self.numcore:
            nc.phase_mag_real_core(dim_uv[0], dim_uv[1], v_phi, u_phi,
                                   v_mag, u_mag, phase, threshold)
        else:
            for j in range(dim_uv[0]):
                for i in range(dim_uv[1]):
                    u_phase = u_phi[dim_uv[0]-1-j:(2*dim_uv[0]-1)-j,
                                    dim_uv[1]-1-i:(2*dim_uv[1]-1)-i]
                    if abs(u_mag[j, i]) > threshold:
                        phase += u_mag[j, i] * u_phase
                    v_phase = v_phi[dim_uv[0]-1-j:(2*dim_uv[0]-1)-j,
                                    dim_uv[1]-1-i:(2*dim_uv[1]-1)-i]
                    if abs(v_mag[j, i]) > threshold:
                        phase -= v_mag[j, i] * v_phase
        # Return the phase:
        return PhaseMap(self.a, phase)

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(a=%r, projector=%r, b_0=%r, threshold=%r, geometry=%r, numcore=%r)' % \
            (self.__class__, self.a, self.projector, self.b_0,
             self.threshold, self.geometry, self.numcore)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'PMReal(a=%s, projector=%s, b_0=%s, threshold=%s, geometry=%s, numcore=%s)' % \
            (self.a, self.projector, self.b_0, self.threshold, self.geometry, self.numcore)

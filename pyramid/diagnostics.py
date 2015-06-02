# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.Diagnostics` class for the calculation of diagnostics of a
specified costfunction for a fixed magnetization distribution."""


import numpy as np
import matplotlib.pyplot as plt

import jutil

from pyramid import fft
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap

import logging


__all__ = ['Diagnostics']


class Diagnostics(object):

    '''Class for calculating diagnostic properties of a specified costfunction.

    For the calculation of diagnostic properties, a costfunction and a magnetization distribution
    are specified at construction. With the :func:`~.set_position`, a position in 3D space can be
    set at which all properties will be calculated. Properties are saved via boolean flags and
    thus, calculation is only done if the position has changed in between. The standard deviation
    and the measurement contribution require the execution of a conjugate gradient solver and can
    take a while for larger problems.

    Attributes
    ----------
    x_rec: :class:`~numpy.ndarray`
        Vectorized magnetization distribution at which the costfunction is evaluated.
    cost: :class:`~.pyramid.costfunction.Costfunction`
        Costfunction for which the diagnostics are calculated.
    max_iter: int, optional
        Maximum number of iterations. Default is 1000.
    fwd_model: :class:`~pyramid.forwardmodel.ForwardModel`
        Forward model used in the costfunction.
    Se_inv : :class:`~numpy.ndarray` (N=2), optional
        Inverted covariance matrix of the measurement errors. The matrix has size `NxN` with N
        being the length of the targetvector y (vectorized phase map information).
    dim: tuple (N=3)
        Dimensions of the 3D magnetic distribution.
    row_idx: int
        Row index of the system matrix corresponding to the current position in 3D space.
    cov_row: :class:`~numpy.ndarray`
        Row of the covariance matrix (``S_a^-1+F'(x_f)^T S_e^-1 F'(x_f)``) which is needed for the
        calculation of the gain and averaging kernel matrizes and which ideally contains the
        variance at position `row_idx` for the current component and position in 3D.
    std: float
        Standard deviation of the chosen component at the current position (calculated when
        needed).
    gain_row: :class:`~numpy.ndarray`
        Row of the gain matrix, which maps differences of phase measurements onto differences in
        the retrieval result of the magnetization distribution(calculated when needed).
    avrg_kern_row: :class:`~numpy.ndarray`
        Row of the averaging kernel matrix (which is ideally the identity matrix), which describes
        the smoothing introduced by the regularization (calculated when needed).
    measure_contribution: :class:`~numpy.ndarray`

    Notes
    -----
        Some properties depend on others, which may require recalculations of these prior
        properties if necessary. The dependencies are ('-->' == 'requires'):
        avrg_kern_row --> gain_row --> std --> m_inv_row
        measure_contribution is independant

    '''

    _log = logging.getLogger(__name__+'.Diagnostics')

    @property
    def cov_row(self):
        if not self._updated_cov_row:
            e_i = fft.zeros(self.cost.n, dtype=fft.FLOAT)
            e_i[self.row_idx] = 1
            row = 2 * jutil.cg.conj_grad_solve(self._A, e_i, P=self._P, max_iter=self.max_iter)
            self._std_row = row
            self._updated_cov_row = True
        return self._std_row

    @property
    def std(self):
        return np.sqrt(self.cov_row[self.row_idx])

    @property
    def gain_row(self):
        if not self._updated_gain_row:
            self._gain_row = self.Se_inv.dot(self.fwd_model.jac_dot(self.x_rec, self.cov_row))
            self._updated_gain_row = True
        return self._gain_row

    @property
    def avrg_kern_row(self):
        if not self._updated_avrg_kern_row:
            self._avrg_kern_row = self.fwd_model.jac_T_dot(self.x_rec, self.gain_row)
            self._updated_avrg_kern_row = True
        return self._avrg_kern_row

    @property
    def measure_contribution(self):
        if not self._updated_measure_contribution:
            cache = self.fwd_model.jac_dot(self.x_rec, fft.ones(self.cost.n, fft.FLOAT))
            cache = self.fwd_model.jac_T_dot(self.x_rec, self.Se_inv.dot(cache))
            mc = 2 * jutil.cg.conj_grad_solve(self._A, cache, P=self._P, max_iter=self.max_iter)
            self._measure_contribution = mc
            self._updated_measure_contribution = True
        return self._measure_contribution

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, pos):
        c, z, y, x = pos
        assert self.mask[z, y, x], 'Position is outside of the provided mask!'
        mask_vec = self.mask.flatten()
        idx_3d = z*self.dim[1]*self.dim[2] + y*self.dim[2] + x
        row_idx = c*np.prod(mask_vec.sum()) + mask_vec[:idx_3d].sum()
        if row_idx != self.row_idx:
            self._pos = pos
            self.row_idx = row_idx
            self._updated_cov_row = False
            self._updated_gain_row = False
            self._updated_avrg_kern_row = False
            self._updated_measure_contribution = False

    def __init__(self, x_rec, cost, max_iter=1000):
        self._log.debug('Calling __init__')
        self.x_rec = x_rec
        self.cost = cost
        self.max_iter = max_iter
        self.fwd_model = cost.fwd_model
        self.Se_inv = cost.Se_inv
        self.dim = cost.data_set.dim
        self.mask = cost.data_set.mask
        self.row_idx = None
        self.pos = (0,) + tuple(np.array(np.where(self.mask))[:, 0])  # first True mask entry
        self._updated_cov_row = False
        self._updated_gain_row = False
        self._updated_avrg_kern_row = False
        self._updated_measure_contribution = False
        self._A = jutil.operator.CostFunctionOperator(self.cost, self.x_rec)
        self._P = jutil.preconditioner.CostFunctionPreconditioner(self.cost, self.x_rec)
        self._log.debug('Creating '+str(self))

    def get_avg_kern_row(self, pos=None):
        '''Get the averaging kernel matrix row represented as a 3D magnetization distribution.

        Parameters
        ----------
        pos: tuple (N=4)
            Position in 3D plus component `(c, z, y, x)`

        Returns
        -------
        mag_data_avg_kern: :class:`~pyramid.magdata.MagData`
            Averaging kernel matrix row represented as a 3D magnetization distribution

        '''
        self._log.debug('Calling get_avg_kern_row')
        if pos is not None:
            self.pos = pos
        mag_data_avg_kern = MagData(self.cost.data_set.a, fft.zeros((3,)+self.dim))
        mag_data_avg_kern.set_vector(self.avrg_kern_row, mask=self.mask)
        return mag_data_avg_kern

    def calculate_averaging(self, pos=None):
        '''Calculate and plot the averaging pixel number at a specified position for x, y or z.

        Parameters
        ----------
        pos: tuple (N=4)
            Position in 3D plus component `(c, z, y, x)`

        Returns
        -------
        px_avrg: float
            The number of pixels over which is approximately averaged. The inverse is the FWHM.
        (x, y, z): tuple of floats
            The magnitude of the averaging kernel summed along two axes (the remaining are x, y, z,
            respectively).

        Notes
        -----
        Uses the :func:`~.get_avg_kern_row' function

        '''
        self._log.debug('Calling calculate_averaging')
        mag_data_avg_kern = self.get_avg_kern_row(pos)
        mag_x, mag_y, mag_z = mag_data_avg_kern.magnitude
        x = mag_x.sum(axis=(0, 1))
        y = mag_y.sum(axis=(0, 2))
        z = mag_z.sum(axis=(1, 2))
        plt.figure()
        plt.axhline(y=0, ls='-', color='k')
        plt.axhline(y=1, ls='-', color='k')
        plt.plot(x, label='x', color='r', marker='o')
        plt.plot(y, label='y', color='g', marker='o')
        plt.plot(z, label='z', color='b', marker='o')
        c = self.pos[0]
        data = [x, y, z][c]
        col = ['r', 'g', 'b'][c]
        i_m = np.argmax(data)  # Index of the maximum
        plt.axhline(y=data[i_m], ls='-', color=col)
        plt.axhline(y=data[i_m]/2, ls='--', color=col)
        # Left side:
        for i in np.arange(i_m-1, -1, -1):
            if data[i] < data[i_m]/2:
                l = (data[i_m]/2-data[i])/(data[i+1]-data[i]) + i
                break
        # Right side:
        for i in np.arange(i_m+1, data.size):
            if data[i] < data[i_m]/2:
                r = (data[i_m]/2-data[i-1])/(data[i]-data[i-1]) + i-1
                break
        # Calculate FWHM:
        fwhm = r - l
        px_avrg = 1 / fwhm
        plt.vlines(x=[l, r], ymin=0, ymax=data[i_m]/2, linestyles=':', color=col)
        # Add legend:
        plt.legend()
        return px_avrg, (x, y, z)

    def get_gain_row_maps(self, pos=None):
        '''Get the gain matrix row represented as a list of 2D (inverse) phase maps.

        Parameters
        ----------
        pos: tuple (N=4)
            Position in 3D plus component `(c, z, y, x)`

        Returns
        -------
        gain_map_list: list of :class:`~pyramid.phasemap.PhaseMap`
            Gain matrix row represented as a list of 2D phase maps

        Notes
        -----
        Note that the produced gain maps define the magnetization change at the current position
        in 3d per phase change at the position of the . Take this into account when plotting the
        maps (1/rad instead of rad).

        '''
        self._log.debug('Calling get_gain_row_maps')
        if pos is not None:
            self.pos = pos
        hp = self.cost.data_set.hook_points
        gain_map_list = []
        for i, projector in enumerate(self.cost.data_set.projectors):
            gain = self.gain_row[hp[i]:hp[i+1]].reshape(projector.dim_uv)
            gain_map_list.append(PhaseMap(self.cost.data_set.a, gain))
        return gain_map_list

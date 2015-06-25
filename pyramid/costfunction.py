# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.Costfunction` class which represents a strategy to calculate
the so called `cost` of a threedimensional magnetization distribution."""


import numpy as np

from pyramid.regularisator import NoneRegularisator

import logging


__all__ = ['Costfunction']


class Costfunction(object):
    '''Class for calculating the cost of a 3D magnetic distributions in relation to 2D phase maps.

    Represents a strategy for the calculation of the `cost` of a 3D magnetic distribution in
    relation to two-dimensional phase maps. The `cost` is a measure for the difference of the
    simulated phase maps from the magnetic distributions to the given set of phase maps and from
    a priori knowledge represented by a :class:`~.Regularisator` object. Furthermore this class
    provides convenient methods for the calculation of the derivative :func:`~.jac` or the product
    with the Hessian matrix :func:`~.hess_dot` of the costfunction, which can be used by
    optimizers. All required data should be given in a :class:`~DataSet` object.

    Attributes
    ----------
    regularisator : :class:`~.Regularisator`
        Regularisator class that's responsible for the regularisation term.
    y : :class:`~numpy.ndarray` (N=1)
        Vector which lists all pixel values of all phase maps one after another.
    fwd_model : :class:`~.ForwardModel`
        The Forward model instance which should be used for the simulation of the phase maps which
        will be compared to `y`.
    Se_inv : :class:`~numpy.ndarray` (N=2), optional
        Inverted covariance matrix of the measurement errors. The matrix has size `NxN` with N
        being the length of the targetvector y (vectorized phase map information).
    m: int
        Size of the image space.
    n: int
        Size of the input space.

    '''

    _log = logging.getLogger(__name__+'.Costfunction')

    def __init__(self, fwd_model, regularisator):
        self._log.debug('Calling __init__')
        self.fwd_model = fwd_model
        self.regularisator = regularisator
        if self.regularisator is None:
            self.regularisator = NoneRegularisator()
        # Extract important information:
        data_set = fwd_model.data_set
        self.y = data_set.phase_vec
        self.n = data_set.n
        self.m = data_set.m
        if data_set.Se_inv is None:
            data_set.set_Se_inv_diag_with_conf()
        self.Se_inv = data_set.Se_inv
        self._log.debug('Created '+str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(fwd_model=%r, regularisator=%r)' % \
            (self.__class__, self.fwd_model, self.regularisator)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'Costfunction(fwd_model=%s, fwd_model=%s, regularisator=%s)' % \
            (self.fwd_model, self.fwd_model, self.regularisator)

    def __call__(self, x):
        delta_y = self.fwd_model(x) - self.y
        self.chisq_m = delta_y.dot(self.Se_inv.dot(delta_y))
        self.chisq_a = self.regularisator(x)
        self.chisq = self.chisq_m + self.chisq_a
        return self.chisq

    def init(self, x):
        '''Initialise the costfunction by calculating the different cost terms.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution, for which the cost is calculated.

        Returns
        -------
        None

        '''
        self._log.debug('Calling init')
        self(x)

    def jac(self, x):
        '''Calculate the derivative of the costfunction for a given magnetization distribution.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution, for which the Jacobi vector is calculated.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Jacobi vector which represents the cost derivative of all voxels of the magnetization.

        '''
        assert len(x) == self.n
        return (2 * self.fwd_model.jac_T_dot(x, self.Se_inv.dot(self.fwd_model(x) - self.y))
                + self.regularisator.jac(x))

    def hess_dot(self, x, vector):
        '''Calculate the product of a `vector` with the Hessian matrix of the costfunction.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution at which the Hessian is calculated. The Hessian
            is constant in this case, thus `x` can be set to None (it is not used int the
            computation). It is implemented for the case that in the future nonlinear problems
            have to be solved.
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution which is multiplied by the Hessian.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the input `vector` with the Hessian matrix of the costfunction.

        '''
        return (2 * self.fwd_model.jac_T_dot(x, self.Se_inv.dot(self.fwd_model.jac_dot(x, vector)))
                + self.regularisator.hess_dot(x, vector))

    def hess_diag(self, _):
        ''' Return the diagonal of the Hessian.

        Parameters
        ----------
        _ : undefined
            Unused input

        '''
        return np.ones(self.n)

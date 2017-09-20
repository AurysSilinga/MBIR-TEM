# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.Costfunction` class which represents a strategy to calculate
the so called `cost` of a three dimensional magnetization distribution."""

import logging

import numpy as np

from pyramid.regularisator import NoneRegularisator

__all__ = ['Costfunction']


class Costfunction(object):
    """Class for calculating the cost of a 3D magnetic distributions in relation to 2D phase maps.

    Represents a strategy for the calculation of the `cost` of a 3D magnetic distribution in
    relation to two-dimensional phase maps. The `cost` is a measure for the difference of the
    simulated phase maps from the magnetic distributions to the given set of phase maps and from
    a prior knowledge represented by a :class:`~.Regularisator` object. Furthermore this class
    provides convenient methods for the calculation of the derivative :func:`~.jac` or the product
    with the Hessian matrix :func:`~.hess_dot` of the costfunction, which can be used by
    optimizers. All required data should be given in a :class:`~DataSet` object.

    Attributes
    ----------
    fwd_model : :class:`~.ForwardModel`
        The Forward model instance which should be used for the simulation of the phase maps which
        will be compared to `y`.
    regularisator : :class:`~.Regularisator`, optional
        Regularisator class that's responsible for the regularisation term. If `None` or none is
        given, no regularisation will be used.
    y : :class:`~numpy.ndarray` (N=1)
        Vector which lists all pixel values of all phase maps one after another.
    m: int
        Size of the image space.
    n: int
        Size of the input space.
    Se_inv : :class:`~numpy.ndarray` (N=2), optional
        Inverted covariance matrix of the measurement errors. The matrix has size `m x m` with m
        being the length of the target vector y.

    """

    _log = logging.getLogger(__name__ + '.Costfunction')

    def __init__(self, fwd_model, regularisator=None, track_cost_iterations=10):
        self._log.debug('Calling __init__')
        self.fwd_model = fwd_model
        if regularisator is None:
            self.regularisator = NoneRegularisator()
        else:
            self.regularisator = regularisator
        # Extract information from fwd_model:
        self.y = self.fwd_model.y
        self.n = self.fwd_model.n
        self.m = self.fwd_model.m
        self.Se_inv = self.fwd_model.Se_inv
        self.chisq_m = []
        self.chisq_a = []
        self.track_cost_iterations = track_cost_iterations
        self.cnt_hess_dot = 0
        self._log.debug('Created ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(fwd_model=%r, regularisator=%r)' % \
               (self.__class__, self.fwd_model, self.regularisator)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'Costfunction(fwd_model=%s, fwd_model=%s, regularisator=%s)' % \
               (self.fwd_model, self.fwd_model, self.regularisator)

    def __call__(self, x):
        self.calculate_costs(x)
        self.chisq = self.chisq_m[-1] + self.chisq_a[-1]
        return self.chisq

    def calculate_costs(self, x):
        # TODO: Docstring!
        delta_y = self.fwd_model(x) - self.y
        self.chisq_m.append(delta_y.dot(self.Se_inv.dot(delta_y)))
        self.chisq_a.append(self.regularisator(x))


    def init(self, x):
        # TODO: Ask Jörn, why this exists!
        """Initialise the costfunction by calculating the different cost terms.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution, for which the cost is calculated.

        Returns
        -------
        None

        """
        self._log.debug('Calling init')
        self(x)

    def jac(self, x):
        """Calculate the derivative of the costfunction for a given magnetization distribution.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution, for which the Jacobi vector is calculated.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Jacobi vector which represents the cost derivative of all voxels of the magnetization.

        """
        assert len(x) == self.n, 'Length of input {} does not match n={}'.format(len(x), self.n)
        return (2 * self.fwd_model.jac_T_dot(x, self.Se_inv.dot(self.fwd_model(x) - self.y))
                + self.regularisator.jac(x))

    def hess_dot(self, x, vector):
        """Calculate the product of a `vector` with the Hessian matrix of the costfunction.

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

        """
        # TODO: Tracking better as decorator function? Useful for other things?
        self.cnt_hess_dot += 1  # TODO: Ask Jörn if this belongs here or in CountingCostFunction!
        if self.track_cost_iterations > 0 and self.cnt_hess_dot % self.track_cost_iterations == 0:
            self.calculate_costs(vector)
            #print(self.cnt_hess_dot, len(self.chisq_a)) # TODO:!!!
        return (2 * self.fwd_model.jac_T_dot(x, self.Se_inv.dot(self.fwd_model.jac_dot(x, vector)))
                + self.regularisator.hess_dot(x, vector))

    def hess_diag(self, _):
        # TODO: needed for preconditioner?
        """ Return the diagonal of the Hessian.

        Parameters
        ----------
        _ : undefined
            Unused input

        """
        return np.ones(self.n)

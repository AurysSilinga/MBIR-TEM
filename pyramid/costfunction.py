# -*- coding: utf-8 -*-
"""This module provides the :class:`~.Costfunction` class which represents a strategy to calculate
the so called `cost` of a threedimensional magnetization distribution."""


import numpy as np

from scipy.sparse.linalg import LinearOperator
from scipy.sparse import eye

from pyramid.forwardmodel import ForwardModel
from pyramid.regularisator import ZeroOrderRegularisator
from pyramid.regularisator import NoneRegularisator
import logging


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
    data_set: :class:`~dataset.DataSet`
        :class:`~dataset.DataSet` object, which stores all information for the cost calculation.
    regularisator : :class:`~.Regularisator`
        Regularisator class that's responsible for the regularisation term.
    regularisator: :class:`~Regularisator`
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

    LOG = logging.getLogger(__name__+'.Costfunction')

    def __init__(self, data_set, regularisator):
        self.LOG.debug('Calling __init__')
        self.data_set = data_set
        self.fwd_model = ForwardModel(data_set)
        self.regularisator = regularisator
        if self.regularisator is None:
            self.regularisator = NoneRegularisator()
        # Extract important information:
        self.y = data_set.phase_vec
        self.Se_inv = data_set.Se_inv
        self.n = data_set.n
        self.m = data_set.m
        self.LOG.debug('Created '+str(self))

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(data_set=%r, fwd_model=%r, regularisator=%r)' % \
            (self.__class__, self.data_set, self.fwd_model, self.regularisator)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'Costfunction(data_set=%s, fwd_model=%s, regularisator=%s)' % \
            (self.data_set, self.fwd_model, self.regularisator)

    def init(self, x):
        self(x)

    def __call__(self, x):
        self.LOG.debug('Calling __call__')
        delta_y = self.fwd_model(x) - self.y
        self.chisq_m = delta_y.dot(self.Se_inv.dot(delta_y))
        self.chisq_a = self.regularisator(x)
        self.chisq = self.chisq_m + self.chisq_a
        return self.chisq

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
        self.LOG.debug('Calling jac')
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
        self.LOG.debug('Calling hess_dot')
        return (2 * self.fwd_model.jac_T_dot(x, self.Se_inv.dot(self.fwd_model.jac_dot(x, vector)))
                + self.regularisator.hess_dot(x, vector))

    def hess_diag(self, _):
        return np.ones(self.n)


class CFAdapterScipyCG(LinearOperator):

    '''Adapter class making the :class:`~.Costfunction` class accessible for scipy cg methods.

    This class provides an adapter for the :class:`~.Costfunction` to be usable with the
    :func:`~.scipy.sparse.linalg.cg` function. the :func:`~.matvec` function is overwritten to
    implement a multiplication with the Hessian of the adapted costfunction. This is used in the
    :func:`~pyramid.reconstruction.optimise_sparse_cg` function of the
    :mod:`~pyramid.reconstruction` module.

    Attributes
    ----------
    cost : :class:`~.Costfunction`
        Costfunction which should be made usable in the :func:`~.scipy.sparse.linalg.cg` function.

    '''
    # TODO: make obsolete!

    LOG = logging.getLogger(__name__+'.CFAdapterScipyCG')

    def __init__(self, cost):
        self.LOG.debug('Calling __init__')
        self.cost = cost

    def matvec(self, vector):
        '''Matrix-vector multiplication with the Hessian of the adapted costfunction.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vector which will be multiplied by the Hessian matrix provided by the adapted
            costfunction.

        '''
        self.LOG.debug('Calling matvec')
        return self.cost.hess_dot(None, vector)

    @property
    def shape(self):
        return (self.cost.data_set.n, self.cost.data_set.n)

    @property
    def dtype(self):
        return np.dtype("d")

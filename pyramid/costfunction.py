# -*- coding: utf-8 -*-
"""This module provides the :class:`~.Costfunction` class which represents a strategy to calculate
the so called `cost` of a threedimensional magnetization distribution."""


# TODO: better names for variables (no uppercase, more than one letter)

# TODO: Regularisation Class?

import numpy as np

from scipy.sparse.linalg import LinearOperator
from scipy.sparse import eye

import logging


class Costfunction:

    '''Class for calculating the cost of a 3D magnetic distributions in relation to 2D phase maps.

    Represents a strategy for the calculation of the `cost` of a 3D magnetic distribution in
    relation to two-dimensional phase maps. The `cost` is a measure for the difference of the
    simulated phase maps from the magnetic distributions to the given set of phase maps.
    Furthermore this class provides convenient methods for the calculation of the derivative
    :func:`~.jac` or the product with the Hessian matrix :func:`~.hess_dot` of the costfunction,
    which can be used by optimizers.

    Attributes
    ----------
    y : :class:`~numpy.ndarray` (N=1)
        Vector which lists all pixel values of all phase maps one after another. Usually gotten
        via the :class:`~.DataSet` classes `phase_vec` property.
    fwd_model : :class:`~.ForwardModel`
        The Forward model instance which should be used for the simulation of the phase maps which
        will be compared to `y`.
    lam : float, optional
        Regularization parameter used in the Hessian matrix multiplication. Default is 0.

    '''

    LOG = logging.getLogger(__name__+'.Costfunction')

    def __init__(self, y, fwd_model, lam=0):
        self.LOG.debug('Calling __init__')
        self.y = y
        self.fwd_model = fwd_model
        self.lam = lam
        self.Se_inv = eye(len(y))
        self.LOG.debug('Created '+str(self))

    def __call__(self, x):
        self.LOG.debug('Calling __call__')
        y = self.y
        F = self.F
        Se_inv = self.Se_inv
        return (F(x)-y).dot(Se_inv.dot(F(x)-y))

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(fwd_model=%r, lam=%r)' % (self.__class__, self.fwd_model, self.lam)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'Costfunction(fwd_model=%s, lam=%s)' % (self.__class__, self.fwd_model, self.lam)

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
        y = self.y
        F = self.fwd_model
        Se_inv = self.Se_inv
        return F.jac_T_dot(x, Se_inv.dot(F(x)-y))

    def hess_dot(self, x, vector):
        '''Calculate the product of a `vector` with the Hession matrix of the costfunction.

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
        F = self.fwd_model
        Se_inv = self.Se_inv
        lam = self.lam
        return F.jac_T_dot(x, Se_inv.dot(F.jac_dot(x, vector))) + lam*vector


class CFAdapterScipyCG(LinearOperator):

    '''Adapter class making the :class:`~.Costfunction` class accessible for scipy cg methods.

    This class provides an adapter for the :class:`~.Costfunction` to be usable with the
    :func:`~.scipy.sparse.linalg.cg` function. the :func:`~.matvec` function is overwritten to
    implement a multiplication with the Hessian of the adapted costfunction. This is used in the
    :func:`~pyramid.reconstruction.optimice_sparse_cg` function of the
    :mod:`~pyramid.reconstruction` module.

    Attributes
    ----------
    cost : :class:`~.Costfunction`
        :class:`~.Costfunction` class which should be made usable in the
        :func:`~.scipy.sparse.linalg.cg` function.

    '''

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
        return (3*self.cost.fwd_model.size_3d, 3*self.cost.fwd_model.size_3d)

    @property
    def dtype(self):
        return np.dtype("d")

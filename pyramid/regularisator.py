# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.Regularisator` class which represents a regularisation term
which adds additional constraints to a costfunction to minimize."""

import abc
import logging

import numpy as np
from scipy import sparse

import jutil.diff as jdiff
import jutil.norms as jnorm

__all__ = ['NoneRegularisator', 'ZeroOrderRegularisator', 'FirstOrderRegularisator',
           'ComboRegularisator']


class Regularisator(object, metaclass=abc.ABCMeta):
    """Class for providing a regularisation term which implements additional constraints.

    Represents a certain constraint for the 3D magnetization distribution whose cost is to minimize
    in addition to the derivation from the 2D phase maps. Important is the used `norm` and the
    regularisation parameter `lam` (lambda) which determines the weighting between the two cost
    parts (measurements and regularisation). Additional parameters at the end of the input
    vector, which are not relevant for the regularisation can be discarded by specifying the
    number in `add_params`.

    Attributes
    ----------
    norm : :class:`~jutil.norm.WeightedNorm`
        Norm, which is used to determine the cost of the regularisation term.
    lam : float
        Regularisation parameter determining the weighting between measurements and regularisation.
    add_params : int
        Number of additional parameters which are not used in the regularisation. Used to cut
        the input vector into the appropriate size.

    """

    _log = logging.getLogger(__name__ + '.Regularisator')

    @abc.abstractmethod
    def __init__(self, norm, lam, add_params=0):
        self._log.debug('Calling __init__')
        self.norm = norm
        self.lam = lam
        self.add_params = add_params
        if self.add_params > 0:
            self.slice = slice(-add_params)
        else:
            self.slice = slice(None)
        self._log.debug('Created ' + str(self))

    def __call__(self, x):
        self._log.debug('Calling __call__')
        return self.lam * self.norm(x[self.slice])

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(norm=%r, lam=%r, add_params=%r)' % (self.__class__, self.norm, self.lam,
                                                       self.add_params)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'Regularisator(norm=%s, lam=%s, add_params=%s)' % (self.norm, self.lam,
                                                                  self.add_params)

    def jac(self, x):
        """Calculate the derivative of the regularisation term for a given magnetic distribution.

        Parameters
        ----------
        x: :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution, for which the Jacobi vector is calculated.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Jacobi vector which represents the cost derivative of all voxels of the magnetization.

        """
        result = np.zeros_like(x)
        result[self.slice] = self.lam * self.norm.jac(x[self.slice])
        return result

    def hess_dot(self, x, vector):
        """Calculate the product of a `vector` with the Hessian matrix of the regularisation term.

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
            Product of the input `vector` with the Hessian matrix.

        """
        result = np.zeros_like(vector)
        result[self.slice] = self.lam * self.norm.hess_dot(x, vector[self.slice])
        return result

    def hess_diag(self, x):
        """ Return the diagonal of the Hessian.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution at which the Hessian is calculated. The Hessian
            is constant in this case, thus `x` can be set to None (it is not used in the
            computation). It is implemented for the case that in the future nonlinear problems
            have to be solved.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Diagonal of the Hessian matrix.

        """
        self._log.debug('Calling hess_diag')
        result = np.zeros_like(x)
        result[self.slice] = self.lam * self.norm.hess_diag(x[self.slice])
        return result


class ComboRegularisator(Regularisator):
    """Class for providing a regularisation term which combines several regularisators.

    If more than one regularisation should be utilized, this class can be use. It is given a list
    of :class:`~.Regularisator` objects. The input will be forwarded to each of them and the
    results are summed up and returned.

    Attributes
    ----------
    reg_list: :class:`~.Regularisator`
        A list of regularisator objects to whom the input is passed on.

    """

    def __init__(self, reg_list):
        self._log.debug('Calling __init__')
        self.reg_list = reg_list
        super().__init__(norm=None, lam=None)
        self._log.debug('Created ' + str(self))

    def __call__(self, x):
        self._log.debug('Calling __call__')
        return np.sum([self.reg_list[i](x) for i in range(len(self.reg_list))], axis=0)

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(reg_list=%r)' % (self.__class__, self.reg_list)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'ComboRegularisator(reg_list=%s)' % self.reg_list

    def jac(self, x):
        """Calculate the derivative of the regularisation term for a given magnetic distribution.

        Parameters
        ----------
        x: :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution, for which the Jacobi vector is calculated.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Jacobi vector which represents the cost derivative of all voxels of the magnetization.

        """
        return np.sum([self.reg_list[i].jac(x) for i in range(len(self.reg_list))], axis=0)

    def hess_dot(self, x, vector):
        """Calculate the product of a `vector` with the Hessian matrix of the regularisation term.

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
            Product of the input `vector` with the Hessian matrix.

        """
        return np.sum([self.reg_list[i].hess_dot(x, vector) for i in range(len(self.reg_list))],
                      axis=0)

    def hess_diag(self, x):
        """ Return the diagonal of the Hessian.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution at which the Hessian is calculated. The Hessian
            is constant in this case, thus `x` can be set to None (it is not used in the
            computation). It is implemented for the case that in the future nonlinear problems
            have to be solved.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Diagonal of the Hessian matrix.

        """
        self._log.debug('Calling hess_diag')
        return np.sum([self.reg_list[i].hess_diag(x) for i in range(len(self.reg_list))], axis=0)


class NoneRegularisator(Regularisator):
    """Placeholder class if no regularization is used.

    This class is instantiated in the :class:`~pyramid.costfunction.Costfunction`, which means
    no regularisation is used. All associated functions return appropriate zero-values.

    Attributes
    ----------
    norm: None
        No regularization is used, thus also no norm.
    lam: 0
        Not used.

    """

    _log = logging.getLogger(__name__ + '.NoneRegularisator')

    def __init__(self):
        self._log.debug('Calling __init__')
        self.norm = None
        self.lam = 0
        self.add_params = None
        super().__init__(norm=None, lam=None)
        self._log.debug('Created ' + str(self))

    def __call__(self, x):
        self._log.debug('Calling __call__')
        return 0

    def jac(self, x):
        """Calculate the derivative of the regularisation term for a given magnetic distribution.

        Parameters
        ----------
        x: :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution, for which the Jacobi vector is calculated.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Jacobi vector which represents the cost derivative of all voxels of the magnetization.

        """
        return np.zeros_like(x)

    def hess_dot(self, x, vector):
        """Calculate the product of a `vector` with the Hessian matrix of the regularisation term.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution at which the Hessian is calculated. The Hessian
            is constant in this case, thus `x` can be set to None (it is not used in the
            computation). It is implemented for the case that in the future nonlinear problems
            have to be solved.
        vector : a :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution which is multiplied by the Hessian.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Product of the input `vector` with the Hessian matrix of the costfunction.

        """
        return np.zeros_like(vector)

    def hess_diag(self, x):
        """ Return the diagonal of the Hessian.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Vectorized magnetization distribution at which the Hessian is calculated. The Hessian
            is constant in this case, thus `x` can be set to None (it is not used int the
            computation). It is implemented for the case that in the future nonlinear problems
            have to be solved.

        Returns
        -------
        result : :class:`~numpy.ndarray` (N=1)
            Diagonal of the Hessian matrix.

        """
        self._log.debug('Calling hess_diag')
        return np.zeros_like(x)


class ZeroOrderRegularisator(Regularisator):
    """Class for providing a regularisation term which implements Lp norm minimization.

    The constraint this class represents is the minimization of the Lp norm for the 3D
    magnetization distribution. Important is the regularisation parameter `lam` (lambda) which
    determines the weighting between the two cost parts (measurements and regularisation).

    Attributes
    ----------
    lam: float
        Regularisation parameter determining the weighting between measurements and regularisation.
    p: int, optional
        Order of the norm (default: 2, which means a standard L2-norm).
    add_params : int
        Number of additional parameters which are not used in the regularisation. Used to cut
        the input vector into the appropriate size.

    """

    _log = logging.getLogger(__name__ + '.ZeroOrderRegularisator')

    def __init__(self, _=None, lam=1E-4, p=2, add_params=0):
        self._log.debug('Calling __init__')
        self.p = p
        if p == 2:
            norm = jnorm.L2Square()
        else:
            norm = jnorm.LPPow(p, 1e-12)
        super().__init__(norm, lam, add_params)
        self._log.debug('Created ' + str(self))


class FirstOrderRegularisator(Regularisator):
    """Class for providing a regularisation term which implements derivation minimization.

    The constraint this class represents is the minimization of the first order derivative of the
    3D magnetization distribution using a Lp norm. Important is the regularisation parameter `lam`
    (lambda) which determines the weighting between the two cost parts (measurements and
    regularisation).

    Attributes
    ----------
    mask: :class:`~numpy.ndarray` (N=3)
        A boolean mask which defines the magnetized volume in 3D.
    lam: float
        Regularisation parameter determining the weighting between measurements and regularisation.
    p: int, optional
        Order of the norm (default: 2, which means a standard L2-norm).
    add_params : int
        Number of additional parameters which are not used in the regularisation. Used to cut
        the input vector into the appropriate size.

    """

    def __init__(self, mask, lam=1E-4, p=2, add_params=0):
        self.p = p
        D0 = jdiff.get_diff_operator(mask, 0, 3)
        D1 = jdiff.get_diff_operator(mask, 1, 3)
        D2 = jdiff.get_diff_operator(mask, 2, 3)
        D = sparse.vstack([D0, D1, D2])
        if p == 2:
            norm = jnorm.WeightedL2Square(D)
        else:
            norm = jnorm.WeightedTV(jnorm.LPPow(p, 1e-12), D, [D0.shape[0], D.shape[0]])
        super().__init__(norm, lam, add_params)
        self._log.debug('Created ' + str(self))

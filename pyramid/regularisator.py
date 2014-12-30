# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 09:24:58 2014

@author: Jan
"""  # TODO: Docstring!


import abc

import numpy as np

import jutil.norms as jnorm
import jutil.diff as jdiff
import jutil.operator as joperator

import logging


__all__ = ['NoneRegularisator', 'ZeroOrderRegularisator', 'FirstOrderRegularisator']

# TODO: Fragen für Jörn: Macht es Sinn, x_a standardmäßig auf den Nullvektor zu setzen? Ansonsten
#       besser im jeweiligen Konstruktor setzen, nicht im abstrakten!
#       Wie kommt man genau an die Ableitungen (Normen sind nicht unproblematisch)?
#       Wie implementiert man am besten verschiedene Normen?


class Regularisator(object):
    # TODO: Docstring!

    __metaclass__ = abc.ABCMeta
    _log = logging.getLogger(__name__+'.Regularisator')

    @abc.abstractmethod
    def __init__(self, norm, lam):
        self._log.debug('Calling __init__')
        self.norm = norm
        self.lam = lam
        self._log.debug('Created '+str(self))

    def __call__(self, x):
        self._log.debug('Calling __call__')
        return self.lam * self.norm(x)

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(norm=%r, lam=%r)' % (self.__class__, self.norm, self.lam)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'Regularisator(norm=%s, lam=%s)' % (self.norm, self.lam)

    def jac(self, x):
        # TODO: Docstring!
        return self.lam * self.norm.jac(x)

    def hess_dot(self, x, vector):
        # TODO: Docstring!
        return self.lam * self.norm.hess_dot(x, vector)

    def hess_diag(self, x, vector):
        # TODO: Docstring!
        self._log.debug('Calling hess_diag')
        return self.lam * self.norm.hess_diag(x, vector)


class NoneRegularisator(Regularisator):
    # TODO: Docstring

    # TODO: Necessary class? Use others with lam=0?

    LOG = logging.getLogger(__name__+'.NoneRegularisator')

    def __init__(self):
        self._log.debug('Calling __init__')
        self.norm = None
        self.lam = 0
        self._log.debug('Created '+str(self))

    def __call__(self, x):
        self._log.debug('Calling __call__')
        return 0

    def jac(self, x):
        # TODO: Docstring!
        return np.zeros_like(x)

    def hess_dot(self, x, vector):
        # TODO: Docstring!
        return np.zeros_like(vector)

    def hess_diag(self, x, vector):
        # TODO: Docstring!
        self._log.debug('Calling hess_diag')
        return np.zeros_like(vector)


class ZeroOrderRegularisator(Regularisator):
    # TODO: Docstring!

    LOG = logging.getLogger(__name__+'.ZeroOrderRegularisator')

    def __init__(self, _, lam, p=2):
        self._log.debug('Calling __init__')
        self.p = p
        if p == 2:
            norm = jnorm.L2Square()
        else:
            norm = jnorm.LPPow(p, 1e-12)
        super(ZeroOrderRegularisator, self).__init__(norm, lam)
        self._log.debug('Created '+str(self))


class FirstOrderRegularisator(Regularisator):
    # TODO: Docstring!

    def __init__(self, mask, lam, p=2):
        self.p = p
        D0 = jdiff.get_diff_operator(mask, 0, 3)
        D1 = jdiff.get_diff_operator(mask, 1, 3)
        D2 = jdiff.get_diff_operator(mask, 2, 3)
        D = joperator.VStack([D0, D1, D2])
        if p == 2:
            norm = jnorm.WeightedL2Square(D)
        else:
            norm = jnorm.WeightedTV(jnorm.LPPow(p, 1e-12), D, [D0.shape[0], D.shape[0]])
        super(FirstOrderRegularisator, self).__init__(norm, lam)
        self._log.debug('Created '+str(self))

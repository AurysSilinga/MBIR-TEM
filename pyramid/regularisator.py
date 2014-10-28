# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 09:24:58 2014

@author: Jan
"""  # TODO: Docstring!


import abc

import numpy as np

from scipy.sparse import coo_matrix, csr_matrix
import jutil.norms as jnorm

from pyramid.converter import IndexConverter

import logging


# TODO: Fragen für Jörn: Macht es Sinn, x_a standardmäßig auf den Nullvektor zu setzen? Ansonsten
#       besser im jeweiligen Konstruktor setzen, nicht im abstrakten!
#       Wie kommt man genau an die Ableitungen (Normen sind nicht unproblematisch)?
#       Wie implementiert man am besten verschiedene Normen?


class Regularisator(object):
    # TODO: Docstring!

    __metaclass__ = abc.ABCMeta
    LOG = logging.getLogger(__name__+'.Regularisator')

    @abc.abstractmethod
    def __init__(self, norm, lam):
        self.LOG.debug('Calling __init__')
        self.norm = norm
        self.lam = lam
        self.LOG.debug('Created '+str(self))

    def __call__(self, x):
        self.LOG.debug('Calling __call__')
        return self.lam * self.norm(x)

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(norm=%r, lam=%r)' % (self.__class__, self.norm, self.lam)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'Regularisator(norm=%s, lam=%s)' % (self.norm, self.lam)

    def jac(self, x):
        # TODO: Docstring!
        self.LOG.debug('Calling jac')
        return self.lam * self.norm.jac_dot(x)

    def hess_dot(self, x, vector):
        # TODO: Docstring!
        self.LOG.debug('Calling hess_dot')
        return self.lam * self.norm.hess_dot(x, vector)

    def hess_diag(self, x, vector):
        # TODO: Docstring!
        self.LOG.debug('Calling hess_diag')
        return self.lam * self.norm.hess_diag(x, vector)


class NoneRegularisator(Regularisator):
    # TODO: Docstring

    # TODO: Necessary class? Use others with lam=0?

    LOG = logging.getLogger(__name__+'.NoneRegularisator')

    def __init__(self):
        self.LOG.debug('Calling __init__')
        self.norm = None
        self.lam = 0
        self.LOG.debug('Created '+str(self))

    def __call__(self, x):
        self.LOG.debug('Calling __call__')
        return 0

    def jac(self, x):
        # TODO: Docstring!
        self.LOG.debug('Calling jac')
        return np.zeros_like(x)

    def hess_dot(self, x, vector):
        # TODO: Docstring!
        self.LOG.debug('Calling hess_dot')
        return np.zeros_like(vector)

    def hess_diag(self, x, vector):
        # TODO: Docstring!
        self.LOG.debug('Calling hess_diag')
        return np.zeros_like(vector)


class ZeroOrderRegularisator(Regularisator):
    # TODO: Docstring!

    LOG = logging.getLogger(__name__+'.ZeroOrderRegularisator')

    def __init__(self, lam):
        self.LOG.debug('Calling __init__')
        norm = jnorm.NormL2Square()
        super(ZeroOrderRegularisator, self).__init__(norm, lam)
        self.LOG.debug('Created '+str(self))


#class FirstOrderRegularisator(Regularisator):
#    # TODO: Docstring!
#
#    def __init__(self, fwd_model, lam, x_a=None):
#        size_3d = fwd_model.size_3d
#        dim = fwd_model.dim
#        converter = IndexConverter(dim)
#        row = []
#        col = []
#        data = []
#
#        for i in range(size_3d):
#            neighbours = converter.find_neighbour_ind(i)
#
#            Sa_inv = csr_matrix(coo_matrix(data, (rows, columns)), shape=(3*size_3d, 3*size_3d))
#
#        term2 = []
#        for i in range(3):
#            component = mag_data[i, ...]
#            for j in range(3):
#                if component.shape[j] > 1:
#                    term2.append(np.diff(component, axis=j).reshape(-1))
#
#        super(FirstOrderRegularisator, self).__init__(Sa_inv, x_a)
#        self.LOG.debug('Created '+str(self))

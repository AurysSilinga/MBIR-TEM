# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 09:24:58 2014

@author: Jan
"""  # TODO: Docstring!


import abc

import numpy as np

from scipy.sparse import eye, coo_matrix, csr_matrix

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
    def __init__(self, Sa_inv, x_a=None):
        self.LOG.debug('Calling __init__')
        self.Sa_sqrt_inv = Sa_sqrt_inv
        if x_a is None:
            x_a = np.zeros(np.shape(Sa_sqrt_inv)[1])
        self.x_a = x_a
        self.LOG.debug('Created '+str(self))

    def __call__(self, x, x_a=None):
        self.LOG.debug('Calling __call__')
        return (x-self.x_a).dot(self.Sa_sqrt_inv.dot(x-self.x_a))

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(Sa_inv=%r, x_a=%r)' % (self.__class__, self.Sa_inv, self.x_a)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'Regularisator(Sa_inv=%s, x_a=%s)' % (self.Sa_inv, self.x_a)

    def jac_dot(self, vector):
        # TODO: Docstring!
        return self.Sa_inv.dot(vector-self.x_a)


class ZeroOrderRegularisator(Regularisator):
    # TODO: Docstring!

    def __init__(self, fwd_model, lam, x_a=None):
        Sa_inv = lam * eye(3*fwd_model.size_3d)
        super(ZeroOrderRegularisator, self).__init__(Sa_inv, x_a)
        self.LOG.debug('Created '+str(self))


class FirstOrderRegularisator(Regularisator):
    # TODO: Docstring!

    def __init__(self, fwd_model, lam, x_a=None):
        size_3d = fwd_model.size_3d
        dim = fwd_model.dim
        converter = IndexConverter(dim)
        row = []
        col = []
        data = []

        for i in range(size_3d):
            neighbours = converter.find_neighbour_ind(i)










            Sa_inv = csr_matrix(coo_matrix(data, (rows, columns)), shape=(3*size_3d, 3*size_3d))




        term2 = []
        for i in range(3):
            component = mag_data[i, ...]
            for j in range(3):
                if component.shape[j] > 1:
                    term2.append(np.diff(component, axis=j).reshape(-1))


        super(FirstOrderRegularisator, self).__init__(Sa_inv, x_a)
        self.LOG.debug('Created '+str(self))

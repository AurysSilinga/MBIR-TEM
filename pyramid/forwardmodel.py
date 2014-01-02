# -*- coding: utf-8 -*-
"""
Created on Mon Jan 06 14:02:18 2014

@author: Jan
"""


import numpy as np

from pyramid.kernel import Kernel
from pyramid.projector import Projector

class ForwardModel:
    # TODO: Docstring!

    @property
    def projectors(self):
        return self._projectors
    
    @projectors.setter
    def projectors(self, projectors):
        assert np.all([isinstance(projector, Projector) for projector in projectors]), \
            'List has to consist of Projector objects!'
        self._projectors = projectors

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, kernel):
        assert isinstance(kernel, Kernel), 'A Kernel object has to be provided!'
        self._kernel = kernel

    def __init__(self,projectors, kernel):
        # TODO: Docstring!
        self.kernel = kernel
        self.b_0 = kernel.b_0
        self.a = kernel.a
        self.dim = kernel.dim
        self.projectors = projectors

    def __call__(self, x):
        # TODO: Docstring!
        result = [self.kernel.jac_dot(projector.jac_dot(x)) for projector in self.projectors]
        return np.reshape(result, -1)

    # TODO: jac_dot ausschreiben!!!

    def jac_dot(self, x, vector):
        # TODO: Docstring! multiplication with the jacobi-matrix (may depend on x)
        return self(vector)  # The jacobi-matrix does not depend on x in a linear problem
    
    def jac_T_dot(self, x, vector):
        # TODO: Docstring! multiplication with the jacobi-matrix (may depend on x)
        # The jacobi-matrix does not depend on x in a linear problem
        result = [projector.jac_T_dot(self.kernel.jac_T_dot(x)) for projector in self.projectors]
        return np.reshape(result, -1)

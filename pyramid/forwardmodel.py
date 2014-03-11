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

    def __init__(self, projectors, kernel):
        # TODO: Docstring!
        self.kernel = kernel
        self.a = kernel.a
        self.dim_uv = kernel.dim_uv
        self.projectors = projectors

    def __call__(self, x):
        # TODO: Docstring!
#        print 'FWD Model - __call__ -  input: ', len(x)
        result = [self.kernel.jac_dot(projector.jac_dot(x)) for projector in self.projectors]
        result = np.reshape(result, -1)
#        print 'FWD Model - __call__ -  output:', len(result)
        return result

    # TODO: jac_dot ausschreiben!!!

    def jac_dot(self, x, vector):
        # TODO: Docstring! multiplication with the jacobi-matrix (may depend on x)
#        print 'FWD Model - jac_dot - input: ', len(vector)
        result = self(vector)
#        print 'FWD Model - jac_dot - output:', len(result)
        return result  # The jacobi-matrix does not depend on x in a linear problem
    
    def jac_T_dot(self, x, vector):
        # TODO: Docstring! multiplication with the jacobi-matrix (may depend on x)
        # The jacobi-matrix does not depend on x in a linear problem
#        print 'FWD Model - jac_T_dot - input: ', len(vector)
        size_3d = self.projectors[0].size_3d
        size_2d = np.prod(self.dim_uv)
        result = np.zeros(3*size_3d)
        for (i, projector) in enumerate(self.projectors):
            result += projector.jac_T_dot(self.kernel.jac_T_dot(vector[i*size_2d:(i+1)*size_2d]))
#        result = [projector.jac_T_dot(self.kernel.jac_T_dot(vector[i*size_2d:(i+1)*size_2d]))
#                  for (i, projector) in enumerate(self.projectors)]
        result = np.reshape(result, -1)
#        print 'FWD Model - jac_T_dot - output:', len(result)
        return result

# -*- coding: utf-8 -*-
"""This module provides the :class:`~.ForwardModel` class which represents a strategy to map a
threedimensional magnetization distribution onto a two-dimensional phase map."""


import numpy as np

from pyramid.kernel import Kernel
from pyramid.projector import Projector

import logging


class ForwardModel(object):

    '''Class for mapping 3D magnetic distributions to 2D phase maps.

    Represents a strategy for the mapping of a 3D magnetic distribution to two-dimensional
    phase maps. Can handle a list of `projectors` of :class:`~.Projector` objects, which describe
    different projection angles, so many phase_maps can be created from one magnetic distribution.

    Attributes
    ----------
    projectors : list of :class:`~.Projector`
        A list of all :class:`~.Projector` objects representing the projection directions.
    kernel : :class:`~.Kernel`
        A kernel which describes the phasemapping of the 2D projected magnetization distribution.
    a : float
        The grid spacing in nm. Extracted from the `kernel`.
    dim : tuple (N=3)
        Dimensions of the 3D magnetic distribution. Extracted from the `projectors` list.
    dim_uv: tuple (N=2)
        Dimensions of the projected grid. Is extracted from the `kernel`.
    size_3d : int
        Number of voxels of the 3-dimensional grid. Extracted from the `projectors` list.
    size_2d : int
        Number of pixels of the 2-dimensional projected grid. Extracted from the `projectors` list.

    '''

    LOG = logging.getLogger(__name__+'.ForwardModel')

    def __init__(self, projectors, kernel):
        self.LOG.debug('Calling __init__')
        assert np.all([isinstance(projector, Projector) for projector in projectors]), \
            'List has to consist of Projector objects!'
        assert isinstance(kernel, Kernel), 'A Kernel object has to be provided!'
        self.kernel = kernel
        self.a = kernel.a
        self.projectors = projectors
        self.dim = self.projectors[0].dim
        self.size_3d = self.projectors[0].size_3d
        self.dim_uv = kernel.dim_uv
        self.size_2d = kernel.size
        self.LOG.debug('Creating '+str(self))

    def __call__(self, x):
        self.LOG.debug('Calling __call__')
        result = [self.kernel(projector(x)) for projector in self.projectors]
        return np.reshape(result, -1)

    def jac_dot(self, x, vector):
        '''Calculate the product of the Jacobi matrix with a given `vector`.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Evaluation point of the jacobi-matrix. The Jacobi matrix is constant for a linear
            problem, thus `x` can be set to None (it is not used int the computation). It is
            implemented for the case that in the future nonlinear problems have to be solved.
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized form of the 3D magnetization distribution. First the `x`, then the `y` and
            lastly the `z` components are listed.

        Returns
        -------
        result_vector : :class:`~numpy.ndarray` (N=1)
            Product of the Jacobi matrix (which is not explicitely calculated) with the input
            `vector`.

        '''
        self.LOG.debug('Calling jac_dot')
        result = [self.kernel.jac_dot(projector.jac_dot(vector)) for projector in self.projectors]
        result = np.reshape(result, -1)
        return result

    def jac_T_dot(self, x, vector):
        ''''Calculate the product of the transposed Jacobi matrix with a given `vector`.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Evaluation point of the jacobi-matrix. The jacobi matrix is constant for a linear
            problem, thus `x` can be set to None (it is not used int the computation). Is used
            for the case that in the future nonlinear problems have to be solved.
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized form of all 2D phase maps one after another in one vector.

        Returns
        -------
        result_vector : :class:`~numpy.ndarray` (N=1)
            Product of the transposed Jacobi matrix (which is not explicitely calculated) with
            the input `vector`.

        '''
        self.LOG.debug('Calling jac_T_dot')
        size_3d = self.projectors[0].size_3d
        size_2d = np.prod(self.dim_uv)
        result = np.zeros(3*size_3d)
        for (i, projector) in enumerate(self.projectors):
            result += projector.jac_T_dot(self.kernel.jac_T_dot(vector[i*size_2d:(i+1)*size_2d]))
        return np.reshape(result, -1)

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(projectors=%r, kernel=%r)' % (self.__class__, self.projectors, self.kernel)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'ForwardModel(%s -> %s, %s projections, kernel=%s)' % \
            (self.dim, self.dim_uv, len(self.projectors), self.kernel)

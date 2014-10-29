# -*- coding: utf-8 -*-
"""This module provides the :class:`~.ForwardModel` class which represents a strategy to map a
threedimensional magnetization distribution onto a two-dimensional phase map."""


import numpy as np

from pyramid.magdata import MagData
import logging


class ForwardModel(object):

    '''Class for mapping 3D magnetic distributions to 2D phase maps.

    Represents a strategy for the mapping of a 3D magnetic distribution to two-dimensional
    phase maps. Can handle a list of `projectors` of :class:`~.Projector` objects, which describe
    different projection angles, so many phase_maps can be created from one magnetic distribution.
    All required data should be given in a :class:`~DataSet` object.

    Attributes
    ----------
    data_set: :class:`~dataset.DataSet`
        :class:`~dataset.DataSet` object, which stores all required information calculation.
    projectors : list of :class:`~.Projector`
        A list of all :class:`~.Projector` objects representing the projection directions.
    kernel : :class:`~.Kernel`
        A kernel which describes the phasemapping of the 2D projected magnetization distribution.
    a : float
        The grid spacing in nm.
    dim : tuple (N=3)
        Dimensions of the 3D magnetic distribution.
    n: int
        Size of the image space. Number of pixels of the 2-dimensional projected grid.
    m: int
        Size of the input space. Number of voxels of the 3-dimensional grid.

    '''

    LOG = logging.getLogger(__name__+'.ForwardModel')

    def __init__(self, data_set):
        self.LOG.debug('Calling __init__')
        self.data_set = data_set
        self.phase_mappers = data_set.phase_mappers
        self.m = data_set.m
        self.n = data_set.n
        self.hook_points = data_set.hook_points
        self.mag_data = MagData(data_set.a, np.zeros((3,)+data_set.dim))
        self.LOG.debug('Creating '+str(self))

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(data_set=%r)' % (self.__class__, self.data_set)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'ForwardModel(data_set=%s)' % (self.data_set)

    def __call__(self, x):
        self.LOG.debug('Calling __call__')
        self.mag_data.magnitude[:] = 0
        self.mag_data.set_vector(x, self.data_set.mask)
        # TODO: Multiprocessing
        result = np.zeros(self.n)
        hp = self.hook_points
        for i, projector in enumerate(self.data_set.projectors):
            phase_map = self.phase_mappers[projector.dim_uv](projector(self.mag_data))
            result[hp[i]:hp[i+1]] = phase_map.phase_vec
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
        self.mag_data.magnitude[:] = 0
        self.mag_data.set_vector(vector, self.data_set.mask)
        result = np.zeros(self.n)
        hp = self.hook_points
        for i, projector in enumerate(self.data_set.projectors):
            mag_vec = self.mag_data.mag_vec
            res = self.phase_mappers[projector.dim_uv].jac_dot(projector.jac_dot(mag_vec))
            result[hp[i]:hp[i+1]] = res.flatten()
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

        result = np.zeros(3*np.prod(self.data_set.dim))
        hp = self.hook_points
        for i, projector in enumerate(self.data_set.projectors):
            vec = vector[hp[i]:hp[i+1]]
            result += projector.jac_T_dot(self.phase_mappers[projector.dim_uv].jac_T_dot(vec))
        self.mag_data.mag_vec = result
        return self.mag_data.get_vector(self.data_set.mask)

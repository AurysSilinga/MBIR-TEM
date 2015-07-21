# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.ForwardModel` class which represents a strategy to map a
threedimensional magnetization distribution onto a two-dimensional phase map."""


import numpy as np

from pyramid.magdata import MagData
from pyramid.ramp import Ramp

import logging


__all__ = ['ForwardModel']


class ForwardModel(object):

    '''Class for mapping 3D magnetic distributions to 2D phase maps.

    Represents a strategy for the mapping of a 3D magnetic distribution to two-dimensional
    phase maps. A :class:`~.DataSet` object is given which is used as input for the model
    (projectors, phase_mappers, etc.). A `ramp_order` can be specified to add polynomial ramps
    to the constructed phase maps (which can also be reconstructed!). A :class:`~.Ramp` class
    object will be constructed accordingly, which also holds all info about the ramps after a
    reconstruction.

    Attributes
    ----------
    data_set: :class:`~dataset.DataSet`
        :class:`~dataset.DataSet` object, which stores all required information calculation.
    ramp_order : int or None (default)
        Polynomial order of the additional phase ramp which will be added to the phase maps.
        All ramp parameters have to be at the end of the input vector and are split automatically.
        Default is None (no ramps are added).
    m: int
        Size of the image space. Number of pixels of the 2-dimensional projected grid.
    n: int
        Size of the input space. Number of voxels of the 3-dimensional grid.

    '''

    _log = logging.getLogger(__name__+'.ForwardModel')

    def __init__(self, data_set, ramp_order=None):
        self._log.debug('Calling __init__')
        self.data_set = data_set
        self.ramp_order = ramp_order
        self.phase_mappers = self.data_set.phase_mappers
        self.ramp = Ramp(self.data_set, self.ramp_order)
        self.m = self.data_set.m
        self.n = self.data_set.n
        if self.ramp.n is not None:  # Additional parameters have to be fitted!
            self.n += self.ramp.n
        self.shape = (self.m, self.n)
        self.hook_points = data_set.hook_points
        self.mag_data = MagData(self.data_set.a, np.zeros((3,)+self.data_set.dim))
        self._log.debug('Creating '+str(self))
# TODO: Multiprocessing! ##########################################################################
#        nprocs = 4
#        self.nprocs = nprocs
#        self.procs = []
#        if nprocs > 1:
#            # Set up processes:
#            for i, projector in enumerate(data_set.projectors):
#                proc_id = i % nprocs  # index of the process
#                phase_id = i//nprocs  # index of the phasemap in the frame of the process
#                print '---'
#                print 'proc_id: ', proc_id
#                print 'phase_id:', phase_id
#                print '---'
#
#        for i in self.data_set.count:
#            projector = self.data_set.projectors[i]
###################################################################################################

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(data_set=%r)' % (self.__class__, self.data_set)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'ForwardModel(data_set=%s)' % (self.data_set)

    def __call__(self, x):
        # Extract ramp parameters if necessary (x will be shortened!):
        x = self.ramp.extract_ramp_params(x)
        # Reset mag_data and fill with vector:
        self.mag_data.magnitude[...] = 0
        self.mag_data.set_vector(x, self.data_set.mask)
        # Simulate all phase maps and create result vector:
        result = np.zeros(self.m)
        hp = self.hook_points
        for i, projector in enumerate(self.data_set.projectors):
            mapper = self.phase_mappers[projector.dim_uv]
            phase_map = mapper(projector(self.mag_data))
            phase_map += self.ramp(i)  # add ramp!
            result[hp[i]:hp[i+1]] = phase_map.phase_vec
        return np.reshape(result, -1)
# TODO: Multiprocessing! ##########################################################################
#        nprocs = 4
#        # Set up processes:
#        for i, projector in enumerate(self.data_set.projectors):
#            proc_id = i % nprocs  # index of the process
#            phase_id = i//nprocs  # index of the phasemap in the frame of the process
#            print 'proc_id: ', proc_id
#            print 'phase_id:', phase_id
#            p = Process(target=worker, args=())
#            p.start()
#
#        for i in self.data_set.count:
#            projector = self.data_set.projectors[i]
###################################################################################################

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
            lastly the `z` components are listed. Ramp parameters are also added at the end if
            necessary.

        Returns
        -------
        result_vector : :class:`~numpy.ndarray` (N=1)
            Product of the Jacobi matrix (which is not explicitely calculated) with the input
            `vector`.

        '''
        # Extract ramp parameters if necessary (vector will be shortened!):
        vector = self.ramp.extract_ramp_params(vector)
        # Reset mag_data and fill with vector:
        self.mag_data.magnitude[...] = 0
        self.mag_data.set_vector(vector, self.data_set.mask)
        # Simulate all phase maps and create result vector:
        result = np.zeros(self.m)
        hp = self.hook_points
        for i, projector in enumerate(self.data_set.projectors):
            mag_vec = self.mag_data.mag_vec
            mapper = self.phase_mappers[projector.dim_uv]
            res = mapper.jac_dot(projector.jac_dot(mag_vec))
            res += self.ramp.jac_dot(i)  # add ramp!
            result[hp[i]:hp[i+1]] = res
        return result

    def _jac_dot_element(self, mag_vec, projector, phasemapper):
            return phasemapper.jac_dot(projector.jac_dot(mag_vec))  # TODO: ???

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
            the input `vector`. If necessary, transposed ramp parameters are concatenated.

        '''
        proj_T_result = np.zeros(3*np.prod(self.data_set.dim))
        hp = self.hook_points
        for i, projector in enumerate(self.data_set.projectors):
            sub_vec = vector[hp[i]:hp[i+1]]
            mapper = self.phase_mappers[projector.dim_uv]
            proj_T_result += projector.jac_T_dot(mapper.jac_T_dot(sub_vec))
        self.mag_data.mag_vec = proj_T_result
        result = self.mag_data.get_vector(self.data_set.mask)
        ramp_params = self.ramp.jac_T_dot(vector)  # calculate ramp_params separately!
        return np.concatenate((result, ramp_params))

# TODO: Multiprocessing! ##########################################################################
#class DistributedForwardModel(ForwardModel):
#
#    def __init__(self, distributed_data_set, ramp_order=None):
#        self.nprocs = distributed_data_set.nprocs
#        self.fwd_models = []
#        for proc_ind in range(self.nprocs):
#            data_set = distributed_data_set[proc_ind]
#            self.fwd_models.append(ForwardModel(data_set))
###################################################################################################

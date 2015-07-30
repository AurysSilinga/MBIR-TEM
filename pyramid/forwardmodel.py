# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.ForwardModel` class which represents a strategy to map a
threedimensional magnetization distribution onto a two-dimensional phase map."""


from __future__ import division

import sys
import numpy as np
import multiprocessing as mp

from pyramid.magdata import MagData
from pyramid.ramp import Ramp
from pyramid.dataset import DataSet

import logging


__all__ = ['ForwardModel', 'DistributedForwardModel']


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

    def finalize(self):
        pass


# TODO: Multiprocessing! ##########################################################################
class DistributedForwardModel(ForwardModel):

    def __init__(self, data_set, ramp_order=None, nprocs=1):
        super(DistributedForwardModel, self).__init__(data_set, ramp_order)
        self.nprocs = nprocs
        img_per_proc = np.ceil(self.data_set.count / self.nprocs).astype(np.int)
        hp = self.data_set.hook_points
        self.proc_hook_points = [0]
#        self.sub_fwd_models = []
        self.pipes = []
        self.processes = []
        for proc_id in range(self.nprocs):
            # 'PARENT Create SubDataSets:'
            sub_data = DataSet(self.data_set.a, self.data_set.dim, self.data_set.b_0,
                               self.data_set.mask, self.data_set.Se_inv)
            # 'PARENT Distribute data to SubDataSets:'
            start = proc_id*img_per_proc
            stop = np.min(((proc_id+1)*img_per_proc, self.data_set.count))
            self.proc_hook_points.append(hp[stop])
            sub_data.phase_maps = self.data_set.phase_maps[start:stop]
            sub_data.projectors = self.data_set.projectors[start:stop]
            # '... PARENT Create SubForwardModel {}'.format(proc_id)
            sub_fwd_model = ForwardModel(sub_data, ramp_order=None)  # ramps handled in master!
            # '... PARENT Create communication pipe {}'.format(proc_id)
            self.pipes.append(mp.Pipe())
            # '... PARENT Create process {}'.format(proc_id)
            p = mp.Process(name='Worker {}'.format(proc_id), target=_worker,
                           args=(sub_fwd_model, self.pipes[proc_id][1]))
            self.processes.append(p)
            # '... PARENT Start process {}'.format(p.name)
            p.start()
        # 'PARENT Finish __init__\n'
        self._log.debug('Creating '+str(self))

#        print 'PARENT Distribute data to SubDataSets:'
#        for count_id in range(self.data_set.count):
#            proc_id = count_id % nprocs
#            print '... PARENT send image {} to subDataSet {}'.format(count_id, proc_id)
#            self.sub_data_sets[proc_id].append(self.data_set.phase_maps[count_id],
#                                               self.data_set.projectors[count_id])
#        print 'PARENT Start up processes:'
#        self.sub_fwd_models = []
#        self.pipes = []
#        self.processes = []
#        for proc_id in range(self.nprocs):
#            print '... PARENT Create SubForwardModel {}'.format(proc_id)
#            self.sub_fwd_models.append(ForwardModel(self.sub_data_sets[proc_id], ramp_order))
#            print '... PARENT Create communication pipe {}'.format(proc_id)
#            self.pipes.append(mp.Pipe())
#            print '... PARENT Create process {}'.format(proc_id)
#            p = mp.Process(name='Worker {}'.format(proc_id), target=_worker,
#                           args=(self.sub_fwd_models[proc_id], self.pipes[proc_id][1]))
#            self.processes.append(p)
#            print '... PARENT Start process {}'.format(p.name)
#            p.start()
#        print 'PARENT Finish __init__\n'
#        self._log.debug('Creating '+str(self))

    def __call__(self, x):
        # Extract ramp parameters if necessary (x will be shortened!):
        # TODO: how to handle here? all in main thread? distribute to sub_x and give to process?
        # TODO: later method of distributed dataset?
        # TODO: After reconstruction is this saved in this ramp? solve with property?
        x = self.ramp.extract_ramp_params(x)
        # Simulate all phase maps and create result vector:
        # 'PARENT Distribute input to processes and start working:'
        for proc_id in range(self.nprocs):
            # TODO: better with Pool()? Can Pool be more persistent by subclassing custom Pool?
            # '... PARENT Send input to process {}'.format(proc_id)
            self.pipes[proc_id][0].send(('__call__', (x,)))
        # TODO: Calculate ramps here? There should be waiting time... Init result with ramps!
        result = np.zeros(self.m)
        hp = self.hook_points
        php = self.proc_hook_points
        # '... PARENT Calculate ramps:'
        if self.ramp_order is not None:
            for i in range(self.data_set.count):
                result[hp[i]:hp[i+1]] += self.ramp(i).phase.ravel()
        # 'PARENT Get process results from the pipes:'
#        sub_results = []
        for proc_id in range(self.nprocs):
            # '... PARENT Retrieve results from process {}'.format(proc_id)
            result[php[proc_id]:php[proc_id+1]] += self.pipes[proc_id][0].recv()
        return result


    def jac_dot(self, x, vector):
        # Extract ramp parameters if necessary (x will be shortened!):
        # TODO: how to handle here? all in main thread? distribute to sub_x and give to process?
        # TODO: later method of distributed dataset?
        # TODO: After reconstruction is this saved in this ramp? solve with property?
        vector = self.ramp.extract_ramp_params(vector)
        # Simulate all phase maps and create result vector:
        # 'PARENT Distribute input to processes and start working:'
        for proc_id in range(self.nprocs):
            # TODO: better with Pool()? Can Pool be more persistent by subclassing custom Pool?
            # '... PARENT Send input to process {}'.format(proc_id)
            self.pipes[proc_id][0].send(('jac_dot', (None, vector)))
        # TODO: Calculate ramps here? There should be waiting time...
        result = np.zeros(self.m)
        hp = self.hook_points
        php = self.proc_hook_points
        # '... PARENT Calculate ramps:'
        if self.ramp_order is not None:
            for i in range(self.data_set.count):
                result[hp[i]:hp[i+1]] += self.ramp.jac_dot(i)
        # 'PARENT Get process results from the pipes:'
        for proc_id in range(self.nprocs):
            # '... PARENT Retrieve results from process {}'.format(proc_id)
            result[php[proc_id]:php[proc_id+1]] += self.pipes[proc_id][0].recv()
        return result
#        for proc_id in range(self.nprocs):
#            # '... PARENT Retrieve results from process {}'.format(proc_id)
#            sub_results.append(self.pipes[proc_id][0].recv())
#
#        # TODO: SORTING!!!! maybe return also the hookpoints?
#        for i in range(self.data_set.count):  # Go over all projections!
#            proc_id = i % self.nprocs
#            j = i // self.nprocs  # index for subresult!
#            result[hp[i]:hp[i+1]] = sub_results[proc_id][hp[j]:hp[j+1]]


    def jac_T_dot(self, x, vector):
        hp = self.hook_points
        php = self.proc_hook_points
        for proc_id in range(self.nprocs):
            sub_vec = vector[php[proc_id]:php[proc_id+1]]
            self.pipes[proc_id][0].send(('jac_T_dot', (None, sub_vec)))

        ramp_params = self.ramp.jac_T_dot(vector)  # calculate ramp_params separately!
        result = np.zeros(3*self.data_set.mask.sum())

        for proc_id in range(self.nprocs):
            sub_vec = vector[php[proc_id]:php[proc_id+1]]
            result += self.pipes[proc_id][0].recv()

        return np.concatenate((result, ramp_params))


        print 'PARENT Distribute input to processes and start working:'
        for proc_id in range(self.nprocs):
            # TODO: better with Pool()? Can Pool be more persistent by subclassing custom Pool?
            print '... PARENT Send input to process {}'.format(proc_id)
            self.pipes[proc_id][0].send(('jac_T_dot', (None, vector)))
        print 'PARENT Get process results from the pipes:'
        sub_results = []
        for proc_id in range(self.nprocs):
            print '... PARENT Retrieve results from process {}'.format(proc_id)
            sub_results.append(self.pipes[proc_id][0].recv())
        result = np.zeros(self.m)
        hp = self.hook_points
        # TODO: Calculate ramps here? There should be waiting time...
        # TODO: SORTING!!!! maybe return also the hookpoints?
        for i in range(self.data_set.count):  # Go over all projections!
            proc_id = i % self.nprocs
            j = i // self.nprocs  # index for subresult!
            result[hp[i]:hp[i+1]] = sub_results[proc_id][hp[j]:hp[j+1]]
        return result



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

    def finalize(self):
        # 'PARENT Finalize processes:'
        for proc_id in range(self.nprocs):
            self.pipes[proc_id][0].send('STOP')
        # 'PARENT Exit the completed processes:'
        for p in self.processes:
            p.join()
        # 'PARENT All processes joint!'


def _worker(fwd_model, pipe):
    # Has to be directly accessible in the module as a function, NOT a method of a class instance!
    # '... {} starting!'.format(mp.current_process().name)
    sys.stdout.flush()
    for method, arguments in iter(pipe.recv, 'STOP'):
        # '... {} processes method {}'.format(mp.current_process().name, method)
        sys.stdout.flush()
        result = getattr(fwd_model, method)(*arguments)
        pipe.send(result)
    # '... ', mp.current_process().name, 'exiting!'
    sys.stdout.flush()




###################################################################################################

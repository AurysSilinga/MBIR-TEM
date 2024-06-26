# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.ForwardModel` class and '~.ForwardModelCharge' class which represent a strategy to
map a three dimensional magnetization and charge distribution onto a two-dimensional phase map."""

import logging
import multiprocessing as mp
import sys

import numpy as np

from pyramid.dataset import DataSet
from pyramid.fielddata import VectorData, ScalarData
from pyramid.ramp import Ramp

__all__ = ['ForwardModel', 'ForwardModelCharge', 'DistributedForwardModel']


# TODO: Ramp should be a forward model itself! Instead of hookpoints, each ForwardModel should
# TODO: keep track of start and end in vector x (defaults: 0 and -1)!
# TODO: Write CombinedForwardModel class!
# TODO: DataSet should be an argument, but should ONLY contain phasemaps!
# TODO: Maybe a list of PhaseMaps is even better and no DataSet class is needed?
# TODO: But what about the convenience functions?
# TODO: PhaseMaps should contain info about their projection direction!
# TODO: The ForwardModel should then setup the projectors accordingly from this info!
# TODO: Se_inv should be a class of its own and should be constructed by the ForwardModel init!
# TODO: Same goes for the mask! Se_inv and mask should contain all constructers and functions!
# TODO: This way, everything is set up and given AFTER all data are collected, because
# TODO: Se_inv and mask can't be set up before...
# TODO: Hook points belong to the forward models (or better the CombinedForwardModel)
# TODO: Maybe have one ForwardModel per image? (maybe not a good idea...?)
# TODO: Build factory convenience functions for constructing CombinedForwardModels!
# TODO: OR implement arithmetic + (combine) and * (concatenate one after the other)?!
# TODO: DistributedForwardModel and CombinedForwardModel could be the same thing?!
# TODO: Build vector class which should now what it contains (stack of PhaseMaps, VectorData,
# TODO: ScalarData, Ramp parameters, or a combination) and according hook points!
# TODO: Vector should be easily convertible to container classes (e.g. stack of PhaseMaps)!

class ForwardModel(object):
    """Class for mapping 3D magnetic distributions to 2D phase maps.

    Represents a strategy for the mapping of a 3D magnetic distribution to two-dimensional
    phase maps. A :class:`~.DataSet` object is given which is used as input for the model
    (projectors, phasemappers, etc.). A `ramp_order` can be specified to add polynomial ramps
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
    y : :class:`~numpy.ndarray` (N=1)
        Vector which lists all pixel values of all phase maps one after another.
    m: int
        Size of the image space. Number of pixels of the 2-dimensional projected grid.
    n: int
        Size of the input space. Number of voxels of the 3-dimensional grid.
    Se_inv : :class:`~numpy.ndarray` (N=2), optional
        Inverted covariance matrix of the measurement errors. The matrix has size `m x m` with m
        being the length of the targetvector y (vectorized phase map information).

    """

    _log = logging.getLogger(__name__ + '.ForwardModel')

    def __init__(self, data_set, ramp_order=None):
        self._log.debug('Calling __init__')
        self.data_set = data_set
        self.ramp_order = ramp_order
        # Extract information from data_set:
        self.phasemappers = self.data_set.phasemappers
        self.y = self.data_set.phase_vec
        self.n = self.data_set.n
        self.m = self.data_set.m
        self.shape = (self.m, self.n)
        self.hook_points = self.data_set.hook_points
        self.Se_inv = self.data_set.Se_inv
        # Create ramp and change n accordingly:
        self.ramp = Ramp(self.data_set, self.ramp_order)
        self.n += self.ramp.n  # ramp.n is 0 if ramp_order is None
        # Create empty MagData object:
        self.magdata = VectorData(self.data_set.a, np.zeros((3,) + self.data_set.dim))
        self._log.debug('Creating ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(data_set=%r)' % (self.__class__, self.data_set)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'ForwardModel(data_set=%s)' % self.data_set

    def __call__(self, x):
        # TODO: Have an extra forward model without the projector part?
        # TODO: Which also corrects for the thickness? Would be nice!
        # Extract ramp parameters if necessary (x will be shortened!):
        x = self.ramp.extract_ramp_params(x)
        # Reset magdata and fill with vector:
        self.magdata.field[...] = 0
        self.magdata.set_vector(x, self.data_set.mask)
        # Simulate all phase maps and create result vector:
        result = np.zeros(self.m)
        hp = self.hook_points
        for i, projector in enumerate(self.data_set.projectors):
            mapper = self.phasemappers[i]
            phasemap = mapper(projector(self.magdata))
            phasemap += self.ramp(i)  # add ramp!
            result[hp[i]:hp[i + 1]] = phasemap.phase_vec
        return np.reshape(result, -1)

    def jac_dot(self, x, vector):
        """Calculate the product of the Jacobi matrix with a given `vector`.

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

        """
        # Extract ramp parameters if necessary (vector will be shortened!):
        vector = self.ramp.extract_ramp_params(vector)
        # Reset magdata and fill with vector:
        self.magdata.field[...] = 0
        self.magdata.set_vector(vector, self.data_set.mask)
        # Simulate all phase maps and create result vector:
        result = np.zeros(self.m)
        hp = self.hook_points
        for i, projector in enumerate(self.data_set.projectors):
            mag_vec = self.magdata.field_vec
            mapper = self.phasemappers[i]
            res = mapper.jac_dot(projector.jac_dot(mag_vec))
            res += self.ramp.jac_dot(i)  # add ramp!
            result[hp[i]:hp[i + 1]] = res
        return result

    def jac_T_dot(self, x, vector):
        """'Calculate the product of the transposed Jacobi matrix with a given `vector`.

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

        """
        proj_T_result = np.zeros(3 * np.prod(self.data_set.dim))
        hp = self.hook_points
        for i, projector in enumerate(self.data_set.projectors):
            sub_vec = vector[hp[i]:hp[i + 1]]
            mapper = self.phasemappers[i]
            proj_T_result += projector.jac_T_dot(mapper.jac_T_dot(sub_vec))
        self.magdata.field_vec = proj_T_result
        result = self.magdata.get_vector(self.data_set.mask)
        ramp_params = self.ramp.jac_T_dot(vector)  # calculate ramp_params separately!
        return np.concatenate((result, ramp_params))

    def finalize(self):
        """'Finalize the processes and let them join the master process (NOT USED HERE!).

        Returns
        -------
        None

        """
        pass


class ForwardModelCharge(object):
    """Class for mapping 3D charge distributions to 2D phase maps.

    Represents a strategy for the mapping of a 3D charge distribution to two-dimensional
    phase maps. A :class:`~.DataSetCharge` object is given which is used as input for the model
    (projectors, phasemappers, etc.). A `ramp_order` can be specified to add polynomial ramps
    to the constructed phase maps (which can also be reconstructed!). A :class:`~.Ramp` class
    object will be constructed accordingly, which also holds all info about the ramps after a
    reconstruction.

    Attributes
    ----------
    data_set: :class:`~dataset.DataSetCharge`
        :class:`~dataset.DataSetCharge` object, which stores all required information calculation.
    ramp_order : int or None (default)
        Polynomial order of the additional phase ramp which will be added to the phase maps.
        All ramp parameters have to be at the end of the input vector and are split automatically.
        Default is None (no ramps are added).
    y : :class:`~numpy.ndarray` (N=1)
        Vector which lists all pixel values of all phase maps one after another.
    m: int
        Size of the image space. Number of pixels of the 2-dimensional projected grid.
    n: int
        Size of the input space. Number of voxels of the 3-dimensional grid.
    Se_inv : :class:`~numpy.ndarray` (N=2), optional
        Inverted covariance matrix of the measurement errors. The matrix has size `m x m` with m
        being the length of the target vector y (vectorized phase map information).

    """

    _log = logging.getLogger(__name__ + '.ForwardModelCharge')

    def __init__(self, data_set, ramp_order=None):
        self._log.debug('Calling __init__')
        self.data_set = data_set
        self.ramp_order = ramp_order
        # Extract information from data_set:
        self.phasemappers = self.data_set.phasemappers
        self.y = self.data_set.phase_vec
        self.n = self.data_set.n
        self.m = self.data_set.m
        self.shape = (self.m, self.n)
        self.hook_points = self.data_set.hook_points
        self.Se_inv = self.data_set.Se_inv
        # Create ramp and change n accordingly:
        self.ramp = Ramp(self.data_set, self.ramp_order)
        self.n += self.ramp.n  # ramp.n is 0 if ramp_order is None
        # Create empty ElecData object:
        self.elecdata = ScalarData(self.data_set.a, np.zeros(self.data_set.dim))
        self._log.debug('Creating ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(data_set=%r)' % (self.__class__, self.data_set)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'ForwardModel(data_set=%s)' % self.data_set

    def __call__(self, x):
        # Extract ramp parameters if necessary (x will be shortened!):
        x = self.ramp.extract_ramp_params(x)
        # Reset elecdata and fill with vector:
        self.elecdata.field[...] = 0
        self.elecdata.set_vector(x, self.data_set.mask)
        # Simulate all phase maps and create result vector:
        result = np.zeros(self.m)
        hp = self.hook_points
        for i, projector in enumerate(self.data_set.projectors):
            mapper = self.phasemappers[i]
            phasemap = mapper(projector(self.elecdata))
            phasemap += self.ramp(i)  # add ramp!
            result[hp[i]:hp[i + 1]] = phasemap.phase_vec
        return np.reshape(result, -1)

    def jac_dot(self, x, vector):
        """Calculate the product of the Jacobi matrix with a given `vector`.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Evaluation point of the jacobi-matrix. The Jacobi matrix is constant for a linear
            problem, thus `x` can be set to None (it is not used int the computation). It is
            implemented for the case that in the future nonlinear problems have to be solved.
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized form of the 3D charge distribution. First the `x`, then the `y` and
            lastly the `z` components are listed. Ramp parameters are also added at the end if
            necessary.

        Returns
        -------
        result_vector : :class:`~numpy.ndarray` (N=1)
            Product of the Jacobi matrix (which is not explicitly calculated) with the input
            `vector`.

        """
        # Extract ramp parameters if necessary (vector will be shortened!):
        vector = self.ramp.extract_ramp_params(vector)
        # Reset elecdata and fill with vector:
        self.elecdata.field[...] = 0
        self.elecdata.set_vector(vector, self.data_set.mask)
        # Simulate all phase maps and create result vector:
        result = np.zeros(self.m)
        hp = self.hook_points
        for i, projector in enumerate(self.data_set.projectors):
            c_vec = self.elecdata.field_vec
            mapper = self.phasemappers[i]
            res = mapper.jac_dot(projector.jac_dot(c_vec))
            res += self.ramp.jac_dot(i)  # add ramp!
            result[hp[i]:hp[i + 1]] = res
        return result

    def jac_T_dot(self, x, vector):
        """'Calculate the product of the transposed Jacobi matrix with a given `vector`.

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
            Product of the transposed Jacobi matrix (which is not explicitly calculated) with
            the input `vector`. If necessary, transposed ramp parameters are concatenated.

        """
        proj_T_result = np.zeros(np.prod(self.data_set.dim))
        hp = self.hook_points
        for i, projector in enumerate(self.data_set.projectors):
            sub_vec = vector[hp[i]:hp[i + 1]]
            mapper = self.phasemappers[i]
            proj_T_result += projector.jac_T_dot(mapper.jac_T_dot(sub_vec))
        self.elecdata.field_vec = proj_T_result
        result = self.elecdata.get_vector(self.data_set.mask)
        ramp_params = self.ramp.jac_T_dot(vector)  # calculate ramp_params separately!
        return np.concatenate((result, ramp_params))

    def finalize(self):
        """'Finalize the processes and let them join the master process (NOT USED HERE!).

        Returns
        -------
        None

        """
        pass


class DistributedForwardModel(ForwardModel):
    """Multiprocessing class for mapping 3D magnetic distributions to 2D phase maps.

    Subclass of the :class:`~.ForwardModel` class which implements multiprocessing strategies
    to speed up the calculations. The interface is the same, internally, the processes and one
    ForwardModel operating on a subset of the DataSet per process are created during construction.
    Ramps are calculated in the main thread. The :func:`~.finalize` method can be used to force
    the processes to join if the class is no longer used.

    Attributes
    ----------
    data_set: :class:`~dataset.DataSet`
        :class:`~dataset.DataSet` object, which stores all required information calculation.
    ramp_order : int or None (default)
        Polynomial order of the additional phase ramp which will be added to the phase maps.
        All ramp parameters have to be at the end of the input vector and are split automatically.
        Default is None (no ramps are added).
    nprocs: int
        Number of processes which should be created. Default is 1 (not recommended). # TODO: <<<!!!

    """

    def __init__(self, data_set, ramp_order=None, nprocs='auto'):
        # Evoke super constructor to set up the normal ForwardModel:
        super().__init__(data_set, ramp_order)
        # Initialize multiprocessing specific stuff:
        mp.log_to_stderr()
        self._log = mp.get_logger()
        if nprocs == 'auto':
            nprocs = mp.cpu_count() - 2  # Use two cores less to reserve cpu for the system.
        self.nprocs = nprocs
        self.pipes = []
        self.processes = []
        self.proc_hook_points = [0]  # Hook points of the processes in the output vector
        hp = self.data_set.hook_points  # Hook points of the images in the output vector
        # Calculate the best distribution of images to the processes:
        proc_img_range = []
        n = self.data_set.count // nprocs  # min items per process
        r = self.data_set.count % nprocs  # remainder items
        start, stop = 0, n + (r != 0)
        for proc_id in range(self.nprocs):
            proc_img_range.append((start, stop))
            if r > 0:  # use up remainder:
                r -= 1
            start = stop
            stop = stop + n + (r != 0)
        # Set up the workers:
        self._log.info('Creating {} processes'.format(self.nprocs))
        for proc_id, (start, stop) in enumerate(proc_img_range):
            # Create SubDataSets:
            sub_data = DataSet(self.data_set.a, self.data_set.dim, self.data_set.b_0,
                               self.data_set.mask, Se_inv=None)  # Se_inv is set later!
            # Distribute data to SubDataSets:
            self.proc_hook_points.append(hp[stop])
            phasemaps = self.data_set.phasemaps[start:stop]
            projectors = self.data_set.projectors[start:stop]
            sub_data.append(phasemaps, projectors)
            # Create SubForwardModel:
            sub_fwd_model = ForwardModel(sub_data, ramp_order=None)  # ramps handled in master!
            # Create communication pipe:
            master_connection, worker_connection = mp.Pipe(duplex=True)  # duplex: both send/recv.!
            self.pipes.append(master_connection)  # Master only needs one end!
            # Create process:
            p = mp.Process(name='Worker {:02d}'.format(proc_id), target=self._worker,
                           args=(sub_fwd_model, worker_connection))
            self.processes.append(p)
            # Start process and close worker pipe end after passing it to worker (and starting it):
            p.start()
            worker_connection.close()  # Close pipe ends in processes that don't need them!
        self._log.debug('Creating ' + str(self))

    def __call__(self, x):
        # Extract ramp parameters if necessary (x will be shortened!):
        x = self.ramp.extract_ramp_params(x)
        # Distribute input to processes and start working:
        for proc_id in range(self.nprocs):
            self.pipes[proc_id].send(('__call__', (x,)))
        # Initialize result vector and shorten hook point names:
        result = np.zeros(self.m)
        hp = self.hook_points
        php = self.proc_hook_points
        # Calculate ramps (if necessary):
        if self.ramp_order is not None:
            for i in range(self.data_set.count):
                result[hp[i]:hp[i + 1]] += self.ramp(i).phase.ravel()
        # Get process results from the pipes:
        for proc_id in range(self.nprocs):
            result[php[proc_id]:php[proc_id + 1]] += self.pipes[proc_id].recv()
        # Return result:
        return result

    def _worker(self, fwd_model, pipe):
        for method, arguments in iter(pipe.recv, 'STOP'):
            # TODO: Properly rethrow Exceptions to master (set to self.exc_info)!
            # TODO: see: https://nedbatchelder.com/blog/200711/rethrowing_exceptions_in_python.html
            sys.stdout.flush()
            result = getattr(fwd_model, method)(*arguments)
            pipe.send(result)
        sys.stdout.flush()
        pipe.close()  # Close worker end of the pipe too, to allow garbage collection!

    def jac_dot(self, x, vector):
        """Calculate the product of the Jacobi matrix with a given `vector`.

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
            Product of the Jacobi matrix (which is not explicitly calculated) with the input
            `vector`.

        """
        # Extract ramp parameters if necessary (x will be shortened!):
        vector = self.ramp.extract_ramp_params(vector)
        # Distribute input to processes and start working:
        for proc_id in range(self.nprocs):
            self.pipes[proc_id].send(('jac_dot', (None, vector)))
        # Initialize result vector and shorten hook point names:
        result = np.zeros(self.m)
        hp = self.hook_points
        php = self.proc_hook_points
        # Calculate ramps (if necessary):
        if self.ramp_order is not None:
            for i in range(self.data_set.count):
                result[hp[i]:hp[i + 1]] += self.ramp.jac_dot(i)
        # Get process results from the pipes:
        for proc_id in range(self.nprocs):
            result[php[proc_id]:php[proc_id + 1]] += self.pipes[proc_id].recv()
        # Return result:
        return result

    def jac_T_dot(self, x, vector):
        """'Calculate the product of the transposed Jacobi matrix with a given `vector`.

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

        """
        php = self.proc_hook_points
        # Distribute input to processes and start working:
        for proc_id in range(self.nprocs):
            sub_vec = vector[php[proc_id]:php[proc_id + 1]]
            self.pipes[proc_id].send(('jac_T_dot', (None, sub_vec)))
        # Calculate ramps:
        ramp_params = self.ramp.jac_T_dot(vector)  # calculate ramp_params separately!
        # Initialize result vector:
        result = np.zeros(3 * self.data_set.mask.sum())
        # Get process results from the pipes:
        for proc_id in range(self.nprocs):
            result += self.pipes[proc_id].recv()
        # Return result:
        return np.concatenate((result, ramp_params))

    def finalize(self):
        """'Finalize the processes and let them join the master process.

        Returns
        -------
        None

        """
        # Finalize processes:
        for proc_id in range(self.nprocs):
            self.pipes[proc_id].send('STOP')
            self.pipes[proc_id].close()
        # Exit the completed processes:
        for p in self.processes:
            p.join()

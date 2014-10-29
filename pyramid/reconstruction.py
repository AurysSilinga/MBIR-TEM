# -*- coding: utf-8 -*-
"""Reconstruct magnetic distributions from given phasemaps.

This module reconstructs 3-dimensional magnetic distributions (as :class:`~pyramid.magdata.MagData`
objects) from a given set of phase maps (represented by :class:`~pyramid.phasemap.PhaseMap`
objects) by using several model based reconstruction algorithms which use the forward model
provided by :mod:`~pyramid.projector` and :mod:`~pyramid.phasemapper` and a priori knowledge of
the distribution.

"""


import numpy as np

from scipy.sparse.linalg import cg
from scipy.optimize import minimize, leastsq

from pyramid.kernel import Kernel
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PhaseMapperRDFC
from pyramid.forwardmodel import ForwardModel
from pyramid.costfunction import Costfunction, CFAdapterScipyCG
from pyramid.magdata import MagData

import logging


LOG = logging.getLogger(__name__)


class PrintIterator(object):

    '''Iterator class which is responsible to give feedback during reconstruction iterations.

    Parameters
    ----------
    cost : :class:`~.Costfunction`
        :class:`~.Costfunction` class for outputting the `cost` of the current magnetization
        distribution. This should decrease per iteration if the algorithm converges and is only
        printed for a `verbosity` of 2.
    verbosity : {0, 1, 2}, optional
        Parameter defining the verbosity of the output. `2` will show the current number of the
        iteration and the cost of the current distribution. `1` will just show the iteration
        number and `0` will prevent output all together.

    Notes
    -----
    Normally this class should not be used by the user and is instantiated whithin the
    :mod:`~.reconstruction` module itself.

    '''

    LOG = logging.getLogger(__name__+'.PrintIterator')

    def __init__(self, cost, verbosity):
        self.LOG.debug('Calling __init__')
        self.cost = cost
        self.verbosity = verbosity
        assert verbosity in {0, 1, 2}, 'verbosity has to be set to 0, 1 or 2!'
        self.iteration = 0
        self.LOG.debug('Created '+str(self))

    def __call__(self, xk):
        self.LOG.debug('Calling __call__')
        if self.verbosity == 0:
            return
        print 'iteration #', self.next(),
        if self.verbosity > 1:
            print 'cost =', self.cost(xk)
        else:
            print ''

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(cost=%r, verbosity=%r)' % (self.__class__, self.cost, self.verbosity)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'PrintIterator(cost=%s, verbosity=%s)' % (self.cost, self.verbosity)

    def next(self):
        self.iteration += 1
        return self.iteration


def optimize_linear(data, regularisator=None, maxiter=1000, verbosity=0):
    '''Reconstruct a three-dimensional magnetic distribution from given phase maps via the
    conjugate gradient optimizaion method :func:`~.scipy.sparse.linalg.cg`.

    Parameters
    ----------
    data : :class:`~.DataSet`
        :class:`~.DataSet` object containing all phase maps in :class:`~.PhaseMap` objects and all
        projection directions in :class:`~.Projector` objects. These provide the essential
        information for the reconstruction.
    Se_inv : :class:`~numpy.ndarray` (N=2), optional
        Inverted covariance matrix of the measurement errors. The matrix has size `NxN` with N
        being the length of the targetvector y (vectorized phase map information). Defaults to
        an appropriate unity matrix if none is provided.
    regularisator : :class:`~.Regularisator`, optional
        Regularisator class that's responsible for the regularisation term. Defaults to zero
        order Tikhonov if none is provided.
    maxiter : int
        Maximum number of iterations.
    verbosity : {0, 1, 2}, optional
        Parameter defining the verposity of the output. `2` will show the current number of the
        iteration and the cost of the current distribution. `1` will just show the iteration
        number and `0` is the default and will prevent output all together.

    Returns
    -------
    mag_data : :class:`~pyramid.magdata.MagData`
        The reconstructed magnetic distribution as a :class:`~.MagData` object.

    '''
    import jutil
    LOG.debug('Calling optimize_sparse_cg')
    # Set up necessary objects:
    fwd_model = ForwardModel(data)
    cost = Costfunction(data, regularisator)
    print cost(np.zeros(cost.n))
    x_opt = jutil.cg.conj_grad_minimize(cost)
    print cost(x_opt)
    # Create and return fitting MagData object:
    mag_opt = MagData(data.a, np.zeros((3,)+data.dim))
    mag_opt.mag_vec = x_opt
    return mag_opt


def optimize_nonlin(data, first_guess=None, regularisator=None):
    '''Reconstruct a three-dimensional magnetic distribution from given phase maps via the
    conjugate gradient optimizaion method :func:`~.scipy.sparse.linalg.cg`.

    Parameters
    ----------
    data : :class:`~.DataSet`
        :class:`~.DataSet` object containing all phase maps in :class:`~.PhaseMap` objects and all
        projection directions in :class:`~.Projector` objects. These provide the essential
        information for the reconstruction.
    verbosity : {2, 1, 0}, optional
        Parameter defining the verposity of the output. `2` is the default and will show the
        current number of the iteration and the cost of the current distribution. `2` will just
        show the iteration number and `0` will prevent output all together.

    Returns
    -------
    mag_data : :class:`~pyramid.magdata.MagData`
        The reconstructed magnetic distribution as a :class:`~.MagData` object.

    '''
    import jutil
    LOG.debug('Calling optimize_cg')
    if first_guess is None:
        first_guess = MagData(data.a, np.zeros((3,)+data.dim))
    x_0 = first_guess.mag_vec
    fwd_model = ForwardModel(data)
    cost = Costfunction(data, regularisator)
    assert len(x_0) == cost.n, (len(x_0), cost.m, cost.n)
    result = jutil.minimizer.minimize(cost, x_0, options={"conv_rel":1e-20}, tol={"max_iteration":1})
    x_opt = result.x
    print cost(x_opt)
    mag_opt = MagData(data.a, np.zeros((3,)+data.dim))
    mag_opt.mag_vec = x_opt
    return mag_opt


def optimize_simple_leastsq(phase_map, mask, b_0=1, lam=1E-4, order=0):
    '''Reconstruct a magnetic distribution for a 2-D problem with known pixel locations.

    Parameters
    ----------
    phase_map : :class:`~pyramid.phasemap.PhaseMap`
        A :class:`~pyramid.phasemap.PhaseMap` object, representing the phase from which to
        reconstruct the magnetic distribution.
    mask : :class:`~numpy.ndarray` (N=3)
        A boolean matrix (or a matrix consisting of ones and zeros), representing the
        positions of the magnetized voxels in 3 dimensions.
    b_0 : float, optional
        The magnetic induction corresponding to a magnetization `M`\ :sub:`0` in T.
        The default is 1.
    lam : float, optional
        The regularisation parameter. Defaults to 1E-4.
    order : int {0, 1}, optional
        order of the regularisation function. Default is 0 for a Tikhonov regularisation of order
        zero. A first order regularisation, which uses the derivative is available with value 1.

    Returns
    -------
    mag_data : :class:`~pyramid.magdata.MagData`
        The reconstructed magnetic distribution as a :class:`~.MagData` object.

    Notes
    -----
    Only works for a single phase_map, if the positions of the magnetized voxels are known and
    for slice thickness of 1 (constraint for the `z`-dimension).

    '''
    # Read in parameters:
    y_m = phase_map.phase_vec  # Measured phase map as a vector
    a = phase_map.a  # Grid spacing
    dim = mask.shape  # Dimensions of the mag. distr.
    count = mask.sum()  # Number of pixels with magnetization
    # Create empty MagData object for the reconstruction:
    mag_data_rec = MagData(a, np.zeros((3,)+dim))

    # Function that returns the phase map for a magnetic configuration x:
    def F(x):
        mag_data_rec.set_vector(x, mask)
        proj = SimpleProjector(dim)
        phase_map = PhaseMapperRDFC(Kernel(a, proj.dim_uv, b_0))(proj(mag_data_rec))
        return phase_map.phase_vec

    # Cost function of order zero which should be minimized:
    def J_0(x_i):
        y_i = F(x_i)
        term1 = (y_i - y_m)
        term2 = lam * x_i
        return np.concatenate([term1, term2])

    # First order cost function which should be minimized:
    def J_1(x_i):
        y_i = F(x_i)
        term1 = (y_i - y_m)
        mag_data = mag_data_rec.magnitude
        term2 = []
        for i in range(3):
            component = mag_data[i, ...]
            for j in range(3):
                if component.shape[j] > 1:
                    term2.append(np.diff(component, axis=j).reshape(-1))

        term2 = lam * np.concatenate(term2)
        return np.concatenate([term1, term2])

    J_DICT = [J_0, J_1]  # list of cost-functions with different regularisations
    # Reconstruct the magnetization components:
    # TODO Use jutil.minimizer.minimize(jutil.costfunction.LeastSquaresCostFunction(J_DICT[order],
    # ...) or a simpler frontend.
    x_rec, _ = leastsq(J_DICT[order], np.zeros(3*count))
    mag_data_rec.set_vector(x_rec, mask)
    return mag_data_rec

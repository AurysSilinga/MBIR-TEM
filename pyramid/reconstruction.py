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

    LOG = logging.getLogger(__name__ + '.PrintIterator')

    def __init__(self, cost, verbosity):
        self.LOG.debug('Calling __init__')
        self.cost = cost
        self.verbosity = verbosity
        assert verbosity in {0, 1, 2}, 'verbosity has to be set to 0, 1 or 2!'
        self.iteration = 0
        self.LOG.debug('Created ' + str(self))

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


def optimize_linear(data, regularisator=None, max_iter=None):
    '''Reconstruct a three-dimensional magnetic distribution from given phase maps via the
    conjugate gradient optimizaion method :func:`~.scipy.sparse.linalg.cg`.
    Blazingly fast for l2-based cost functions.

    Parameters
    ----------
    data : :class:`~.DataSet`
        :class:`~.DataSet` object containing all phase maps in :class:`~.PhaseMap` objects and all
        projection directions in :class:`~.Projector` objects. These provide the essential
        information for the reconstruction.
    regularisator : :class:`~.Regularisator`, optional
        Regularisator class that's responsible for the regularisation term. Defaults to zero
        order Tikhonov if none is provided.

    Returns
    -------
    mag_data : :class:`~pyramid.magdata.MagData`
        The reconstructed magnetic distribution as a :class:`~.MagData` object.

    '''
    import jutil.cg as jcg
    LOG.debug('Calling optimize_linear')
    # Set up necessary objects:
    cost = Costfunction(data, regularisator)
    LOG.info('Cost before optimization: {}'.format(cost(np.zeros(cost.n))))
    x_opt = jcg.conj_grad_minimize(cost, max_iter=max_iter)
    LOG.info('Cost after optimization: {}'.format(cost(x_opt)))
    # Create and return fitting MagData object:
    mag_opt = MagData(data.a, np.zeros((3,) + data.dim))
    mag_opt.set_vector(x_opt, data.mask)
    return mag_opt


def optimize_nonlin(data, first_guess=None, regularisator=None):
    '''Reconstruct a three-dimensional magnetic distribution from given phase maps via
    steepest descent method. This is slow, but works best for non l2-regularisators.


    Parameters
    ----------
    data : :class:`~.DataSet`
        :class:`~.DataSet` object containing all phase maps in :class:`~.PhaseMap` objects and all
        projection directions in :class:`~.Projector` objects. These provide the essential
        information for the reconstruction.
    first_fuess : :class:`~pyramid.magdata.MagData`
        magnetization to start the non-linear iteration with.
    regularisator : :class:`~.Regularisator`, optional
        Regularisator class that's responsible for the regularisation term.

    Returns
    -------
    mag_data : :class:`~pyramid.magdata.MagData`
        The reconstructed magnetic distribution as a :class:`~.MagData` object.

    '''
    import jutil.minimizer as jmin
    import jutil.norms as jnorms
    LOG.debug('Calling optimize_nonlin')
    if first_guess is None:
        first_guess = MagData(data.a, np.zeros((3,) + data.dim))

    x_0 = first_guess.get_vector(data.mask)
    cost = Costfunction(data, regularisator)
    assert len(x_0) == cost.n, (len(x_0), cost.m, cost.n)

    p = regularisator.p
    q = 1. / (1. - (1. / p))
    lp = regularisator.norm
    lq = jnorms.LPPow(q, 1e-20)

    def preconditioner(_, direc):
        direc_p = direc / abs(direc).max()
        direc_p = 10 * (1. / q) * lq.jac(direc_p)
        return direc_p

    # This Method is semi-best for Lp type problems. Takes forever, though
    LOG.info('Cost before optimization: {}'.format(cost(np.zeros(cost.n))))
    result = jmin.minimize(
        cost, x_0,
        method="SteepestDescent",
        options={"preconditioner": preconditioner},
        tol={"max_iteration": 10000})
    x_opt = result.x
    LOG.info('Cost after optimization: {}'.format(cost(x_opt)))
    mag_opt = MagData(data.a, np.zeros((3,) + data.dim))
    mag_opt.set_vector(x_opt, data.mask)
    return mag_opt


def optimize_splitbregman(data, weight, lam, mu):
    '''
    Reconstructs magnet distribution from phase image measurements using a split bregman
    algorithm with a dedicated TV-l1 norm. Very dedicated, frickle, brittle, and difficult
    to get to work, but fastest option available if it works.

    Seems to work for some 2D examples with weight=lam=1 and mu in [1, .., 1e4].

    Parameters
    ----------
    data : :class:`~.DataSet`
        :class:`~.DataSet` object containing all phase maps in :class:`~.PhaseMap` objects and all
        projection directions in :class:`~.Projector` objects. These provide the essential
        information for the reconstruction.
    weight : float
        Obscure split bregman parameter
    lam : float
        Cryptic split bregman parameter
    mu : float
        flabberghasting split bregman paramter

    Returns
    -------
    mag_data : :class:`~pyramid.magdata.MagData`
        The reconstructed magnetic distribution as a :class:`~.MagData` object.

    '''
    import jutil.splitbregman as jsb
    import jutil.operator as joperator
    import jutil.diff as jdiff
    from pyramid.regularisator import FirstOrderRegularisator
    LOG.debug('Calling optimize_splitbregman')

    # regularisator is actually not necessary, but this makes the cost
    # function to that which is supposedly optimized by split bregman.
    # Thus cost can be used to verify convergence
    regularisator = FirstOrderRegularisator(data.mask, lam / mu, 1)
    x_0 = MagData(data.a, np.zeros((3,) + data.dim)).get_vector(data.mask)
    cost = Costfunction(data, regularisator)
    fwd_mod = cost.fwd_model

    A = joperator.Function(
        (cost.m, cost.n),
        lambda x: fwd_mod.jac_dot(None, x),
        FT=lambda x: fwd_mod.jac_T_dot(None, x))
    D = joperator.VStack([
        jdiff.get_diff_operator(data.mask, 0, 3),
        jdiff.get_diff_operator(data.mask, 1, 3)])
    y = np.asarray(cost.y, dtype=np.double)

    x_opt = jsb.split_bregman_2d(
        A, D, y,
        weight=weight, mu=mu, lambd=lam, max_iter=1000)

    mag_opt = MagData(data.a, np.zeros((3,) + data.dim))
    mag_opt.set_vector(x_opt, data.mask)
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
    mag_data_rec = MagData(a, np.zeros((3,) + dim))

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
    x_rec, _ = leastsq(J_DICT[order], np.zeros(3 * count))
    mag_data_rec.set_vector(x_rec, mask)
    return mag_data_rec

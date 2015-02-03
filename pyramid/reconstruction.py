# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Reconstruct magnetic distributions from given phasemaps.

This module reconstructs 3-dimensional magnetic distributions (as :class:`~pyramid.magdata.MagData`
objects) from a given set of phase maps (represented by :class:`~pyramid.phasemap.PhaseMap`
objects) by using several model based reconstruction algorithms which use the forward model
provided by :mod:`~pyramid.projector` and :mod:`~pyramid.phasemapper` and a priori knowledge of
the distribution.

"""


import numpy as np

from pyramid.costfunction import Costfunction
from pyramid.magdata import MagData

import logging


__all__ = ['optimize_linear', 'optimize_nonlin', 'optimize_splitbregman']
_log = logging.getLogger(__name__)


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
    _log.debug('Calling optimize_linear')
    # Set up necessary objects:
    cost = Costfunction(data, regularisator)
    _log.info('Cost before optimization: {}'.format(cost(np.zeros(cost.n))))
    x_opt = jcg.conj_grad_minimize(cost, max_iter=max_iter).x
    _log.info('Cost after optimization: {}'.format(cost(x_opt)))
    # Create and return fitting MagData object:
    mag_opt = MagData(data.a, np.zeros((3,) + data.dim))
    mag_opt.set_vector(x_opt, data.mask)
    return mag_opt, cost


def optimize_nonlin(data, first_guess=None, regularisator=None):
    '''Reconstruct a three-dimensional magnetic distribution from given phase maps via
    steepest descent method. This is slow, but works best for non l2-regularisators.


    Parameters
    ----------
    data : :class:`~.DataSet`
        :class:`~.DataSet` object containing all phase maps in :class:`~.PhaseMap` objects and all
        projection directions in :class:`~.Projector` objects. These provide the essential
        information for the reconstruction.
    first_guess : :class:`~pyramid.magdata.MagData`
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
    _log.debug('Calling optimize_nonlin')
    if first_guess is None:
        first_guess = MagData(data.a, np.zeros((3,) + data.dim))

    x_0 = first_guess.get_vector(data.mask)
    cost = Costfunction(data, regularisator)
    assert len(x_0) == cost.n, (len(x_0), cost.m, cost.n)

    p = regularisator.p
    q = 1. / (1. - (1. / p))
    lq = jnorms.LPPow(q, 1e-20)

    def preconditioner(_, direc):
        direc_p = direc / abs(direc).max()
        direc_p = 10 * (1. / q) * lq.jac(direc_p)
        return direc_p

    # This Method is semi-best for Lp type problems. Takes forever, though
    _log.info('Cost before optimization: {}'.format(cost(np.zeros(cost.n))))
    result = jmin.minimize(
        cost, x_0,
        method="SteepestDescent",
        options={"preconditioner": preconditioner},
        tol={"max_iteration": 10000})
    x_opt = result.x
    _log.info('Cost after optimization: {}'.format(cost(x_opt)))
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
    _log.debug('Calling optimize_splitbregman')

    # regularisator is actually not necessary, but this makes the cost
    # function to that which is supposedly optimized by split bregman.
    # Thus cost can be used to verify convergence
    regularisator = FirstOrderRegularisator(data.mask, lam / mu, 1)
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

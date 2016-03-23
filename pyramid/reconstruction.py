# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Reconstruct magnetic distributions from given phasemaps.

This module reconstructs 3-dimensional magnetic distributions (as
:class:`~pyramid.magdata.VectorData` objects) from a given set of phase maps (represented by
:class:`~pyramid.phasemap.PhaseMap` objects) by using several model based reconstruction algorithms
 which use the forward model provided by :mod:`~pyramid.projector` and :mod:`~pyramid.phasemapper`
 and a priori knowledge of the distribution.

"""

import logging

import numpy as np

from pyramid.fielddata import VectorData

__all__ = ['optimize_linear', 'optimize_nonlin', 'optimize_splitbregman']
_log = logging.getLogger(__name__)


def optimize_linear(costfunction, max_iter=None):
    """Reconstruct a three-dimensional magnetic distribution from given phase maps via the
    conjugate gradient optimizaion method :func:`~.scipy.sparse.linalg.cg`.
    Blazingly fast for l2-based cost functions.

    Parameters
    ----------
    costfunction : :class:`~.Costfunction`
        A :class:`~.Costfunction` object which implements a specified forward model and
        regularisator which is minimized in the optimization process.
    max_iter : int, optional
        The maximum number of iterations for the opimization.

    Returns
    -------
    mag_data : :class:`~pyramid.fielddata.VectorData`
        The reconstructed magnetic distribution as a :class:`~.VectorData` object.

    """
    import jutil.cg as jcg
    _log.debug('Calling optimize_linear')
    _log.info('Cost before optimization: {}'.format(costfunction(np.zeros(costfunction.n))))
    x_opt = jcg.conj_grad_minimize(costfunction, max_iter=max_iter).x
    _log.info('Cost after optimization: {}'.format(costfunction(x_opt)))
    # Cut ramp parameters if necessary (this also saves the final parameters in the ramp class!):
    x_opt = costfunction.fwd_model.ramp.extract_ramp_params(x_opt)
    # Create and return fitting VectorData object:
    data_set = costfunction.fwd_model.data_set
    mag_opt = VectorData(data_set.a, np.zeros((3,) + data_set.dim))
    mag_opt.set_vector(x_opt, data_set.mask)
    return mag_opt


def optimize_nonlin(costfunction, first_guess=None):
    """Reconstruct a three-dimensional magnetic distribution from given phase maps via
    steepest descent method. This is slow, but works best for non l2-regularisators.


    Parameters
    ----------
    costfunction : :class:`~.Costfunction`
        A :class:`~.Costfunction` object which implements a specified forward model and
        regularisator which is minimized in the optimization process.
    first_guess : :class:`~pyramid.fielddata.VectorData`
        magnetization to start the non-linear iteration with.

    Returns
    -------
    mag_data : :class:`~pyramid.fielddata.VectorData`
        The reconstructed magnetic distribution as a :class:`~.VectorData` object.

    """
    import jutil.minimizer as jmin
    import jutil.norms as jnorms
    _log.debug('Calling optimize_nonlin')
    data_set = costfunction.fwd_model.data_set
    if first_guess is None:
        first_guess = VectorData(data_set.a, np.zeros((3,) + data_set.dim))

    x_0 = first_guess.get_vector(data_set.mask)
    assert len(x_0) == costfunction.n, (len(x_0), costfunction.m, costfunction.n)

    p = costfunction.regularisator.p
    q = 1. / (1. - (1. / p))
    lq = jnorms.LPPow(q, 1e-20)

    def _preconditioner(_, direc):
        direc_p = direc / abs(direc).max()
        direc_p = 10 * (1. / q) * lq.jac(direc_p)
        return direc_p

    # This Method is semi-best for Lp type problems. Takes forever, though
    _log.info('Cost before optimization: {}'.format(costfunction(np.zeros(costfunction.n))))
    result = jmin.minimize(
        costfunction, x_0,
        method="SteepestDescent",
        options={"preconditioner": _preconditioner},
        tol={"max_iteration": 10000})
    x_opt = result.x
    _log.info('Cost after optimization: {}'.format(costfunction(x_opt)))
    mag_opt = VectorData(data_set.a, np.zeros((3,) + data_set.dim))
    mag_opt.set_vector(x_opt, data_set.mask)
    return mag_opt


def optimize_splitbregman(costfunction, weight, lam, mu):
    """
    Reconstructs magnet distribution from phase image measurements using a split bregman
    algorithm with a dedicated TV-l1 norm. Very dedicated, frickle, brittle, and difficult
    to get to work, but fastest option available if it works.

    Seems to work for some 2D examples with weight=lam=1 and mu in [1, .., 1e4].

    Parameters
    ----------
    costfunction : :class:`~.Costfunction`
        A :class:`~.Costfunction` object which implements a specified forward model and
        regularisator which is minimized in the optimization process.
    weight : float
        Obscure split bregman parameter
    lam : float
        Cryptic split bregman parameter
    mu : float
        flabberghasting split bregman paramter

    Returns
    -------
    mag_data : :class:`~pyramid.fielddata.VectorData`
        The reconstructed magnetic distribution as a :class:`~.VectorData` object.

    """
    import jutil.splitbregman as jsb
    import jutil.operator as joperator
    import jutil.diff as jdiff
    _log.debug('Calling optimize_splitbregman')

    # regularisator is actually not necessary, but this makes the cost
    # function to that which is supposedly optimized by split bregman.
    # Thus cost can be used to verify convergence
    fwd_model = costfunction.fwd_model
    data_set = fwd_model.data_set

    A = joperator.Function(
        (costfunction.m, costfunction.n),
        lambda x: fwd_model.jac_dot(None, x),
        FT=lambda x: fwd_model.jac_T_dot(None, x))
    D = joperator.VStack([
        jdiff.get_diff_operator(data_set.mask, 0, 3),
        jdiff.get_diff_operator(data_set.mask, 1, 3)])
    y = np.asarray(costfunction.y, dtype=np.double)

    x_opt = jsb.split_bregman_2d(
        A, D, y,
        weight=weight, mu=mu, lambd=lam, max_iter=1000)

    mag_opt = VectorData(data_set.a, np.zeros((3,) + data_set.dim))
    mag_opt.set_vector(x_opt, data_set.mask)
    return mag_opt

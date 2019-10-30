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

from pyramid.fielddata import VectorData, ScalarData

__all__ = ['optimize_linear', 'optimize_linear_charge', 'optimize_nonlin', 'optimize_splitbregman']
_log = logging.getLogger(__name__)


def optimize_linear(costfunction, mag_0=None, ramp_0=None, max_iter=None, verbose=False):
    """Reconstruct a three-dimensional magnetic distribution from given phase maps via the
    conjugate gradient optimization method :func:`~.scipy.sparse.linalg.cg`.
    Blazingly fast for l2-based cost functions.

    Parameters
    ----------
    costfunction : :class:`~.Costfunction`
        A :class:`~.Costfunction` object which implements a specified forward model and
        regularisator which is minimized in the optimization process.
    mag_0: :class:`~.VectorData`
        The starting magnetisation distribution used for the reconstruction. A zero vector will be
        used if no VectorData object is specified.
    ramp_0: :class:`~.Ramp`
        The starting ramp for the reconstruction. A zero vector will be
        used if no Ramp object is specified.
    max_iter : int, optional
        The maximum number of iterations for the optimization.
    verbose: bool, optional
        If set to True, information like a progressbar is displayed during reconstruction.
        The default is False.

    Returns
    -------
    magdata : :class:`~pyramid.fielddata.VectorData`
        The reconstructed magnetic distribution as a :class:`~.VectorData` object.

    """
    import jutil.cg as jcg
    from jutil.taketime import TakeTime
    _log.debug('Calling optimize_linear')
    _log.info('Cost before optimization: {:.3e}'.format(costfunction(np.zeros(costfunction.n))))
    data_set = costfunction.fwd_model.data_set
    # Get starting distribution vector x_0:
    x_0 = np.empty(costfunction.n)
    if mag_0 is not None:
        costfunction.fwd_model.magdata = mag_0
    x_0[:data_set.n] = costfunction.fwd_model.magdata.get_vector(mask=data_set.mask)
    if ramp_0 is not None:
        ramp_vec = ramp_0.param_cache.ravel()
    else:
        ramp_vec = np.zeros_like(costfunction.fwd_model.ramp.n)
    x_0[data_set.n:] = ramp_vec
    # Minimize:
    with TakeTime('reconstruction time'):
        x_opt = jcg.conj_grad_minimize(costfunction, x_0=x_0, max_iter=max_iter, verbose=verbose).x
    _log.info('Cost after optimization: {:.3e}'.format(costfunction(x_opt)))
    # Cut ramp parameters if necessary (this also saves the final parameters in the ramp class!):
    x_opt = costfunction.fwd_model.ramp.extract_ramp_params(x_opt)
    # Create and return fitting VectorData object:
    mag_opt = VectorData(data_set.a, np.zeros((3,) + data_set.dim))
    mag_opt.set_vector(x_opt, data_set.mask)
    return mag_opt


def optimize_linear_charge(costfunction, charge_0=None, ramp_0=None, max_iter=None, verbose=False):
    """Reconstruct a three-dimensional charge distribution from given phase maps via the
    conjugate gradient optimization method :func:`~.scipy.sparse.linalg.cg`.
    Blazingly fast for l2-based cost functions.

    Parameters
    ----------
    costfunction : :class:`~.Costfunction`
        A :class:`~.Costfunction` object which implements a specified forward model and
        regularisator which is minimized in the optimization process.
    charge_0: :class:`~.ScalarData`
        The starting charge distribution used for the reconstruction. A zero vector will be
        used if no ScalarData object is specified.
    ramp_0: :class:`~.Ramp`
        The starting ramp for the reconstruction. A zero vector will be
        used if no Ramp object is specified.
    max_iter : int, optional
        The maximum number of iterations for the optimization.
    verbose: bool, optional
        If set to True, information like a progressbar is displayed during reconstruction.
        The default is False.

    Returns
    -------
    elecdata : :class:`~pyramid.fielddata.ScalarData`
        The reconstructed charge distribution as a :class:`~.ScalarData` object.

    """
    import jutil.cg as jcg
    from jutil.taketime import TakeTime
    _log.debug('Calling optimize_linear_charge')
    _log.info('Cost before optimization: {:.3e}'.format(costfunction(np.zeros(costfunction.n))))
    data_set = costfunction.fwd_model.data_set
    # Get starting distribution vector x_0:
    x_0 = np.empty(costfunction.n)
    if charge_0 is not None:
        costfunction.fwd_model.elecdata = charge_0
    x_0[:data_set.n] = costfunction.fwd_model.elecdata.get_vector(mask=data_set.mask)
    if ramp_0 is not None:
        ramp_vec = ramp_0.param_cache.ravel()
    else:
        ramp_vec = np.zeros_like(costfunction.fwd_model.ramp.n)
    x_0[data_set.n:] = ramp_vec
    # Minimize:
    with TakeTime('reconstruction time'):
        x_opt = jcg.conj_grad_minimize(costfunction, x_0=x_0, max_iter=max_iter, verbose=verbose).x
    _log.info('Cost after optimization: {:.3e}'.format(costfunction(x_opt)))
    # Cut ramp parameters if necessary (this also saves the final parameters in the ramp class!):
    x_opt = costfunction.fwd_model.ramp.extract_ramp_params(x_opt)
    # Create and return fitting ScalarData object:
    charge_opt = ScalarData(data_set.a, np.zeros(data_set.dim))
    charge_opt.set_vector(x_opt, data_set.mask)
    return charge_opt


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
    magdata : :class:`~pyramid.fielddata.VectorData`
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
    magdata : :class:`~pyramid.fielddata.VectorData`
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

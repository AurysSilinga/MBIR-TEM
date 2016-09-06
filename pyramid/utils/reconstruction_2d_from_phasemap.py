# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Reconstruct a magnetization distributions from a single phase map."""

import logging

import numpy as np

from jutil.taketime import TakeTime

from .. import reconstruction
from ..dataset import DataSet
from ..projector import SimpleProjector
from ..regularisator import FirstOrderRegularisator
from ..forwardmodel import ForwardModel
from ..costfunction import Costfunction
from .pm import pm

__all__ = ['reconstruction_2d_from_phasemap']
_log = logging.getLogger(__name__)


def reconstruction_2d_from_phasemap(phasemap, b_0=1, lam=1E-3, max_iter=100, ramp_order=1,
                                    plot_results=False, ar_dens=None):
    """Convenience function for reconstructing a projected distribution from a single phasemap.

    Parameters
    ----------
    phasemap: :class:`~PhaseMap`
        The phasemap which is used for the reconstruction.
    b_0 : float, optional
        The magnetic induction corresponding to a magnetization `M`\ :sub:`0` in T.
        The default is 1.
    lam : float
        Regularisation parameter determining the weighting between measurements and regularisation.
    max_iter : int, optional
        The maximum number of iterations for the opimization.
    ramp_order : int or None (default)
        Polynomial order of the additional phase ramp which will be added to the phase maps.
        All ramp parameters have to be at the end of the input vector and are split automatically.
        Default is None (no ramps are added).
    plot_results: boolean, optional
        If True, the results are plotted after reconstruction.
    ar_dens: int, optional
        Number defining the arrow density which is plotted. A higher ar_dens number skips more
        arrows (a number of 2 plots every second arrow). Default is 1.

    Returns
    -------
    magdata_rec, cost: :class:`~.VectorData`, :class:`~.Costfunction`
        The reconstructed magnetisation distribution and the used costfunction.

    """
    _log.debug('Calling reconstruction_2d_from_phasemap')
    # Construct DataSet, Regularisator, ForwardModel and Costfunction:
    dim = (1,) + phasemap.dim_uv
    data = DataSet(phasemap.a, dim, b_0)
    data.append(phasemap, SimpleProjector(dim))
    data.set_3d_mask()
    fwd_model = ForwardModel(data, ramp_order)
    reg = FirstOrderRegularisator(data.mask, lam, add_params=fwd_model.ramp.n)
    cost = Costfunction(fwd_model, reg)
    # Reconstruct:
    with TakeTime('reconstruction time'):
        magdata_rec = reconstruction.optimize_linear(cost, max_iter=max_iter)
        param_cache = cost.fwd_model.ramp.param_cache
    if ramp_order is None:
        offset, ramp = 0, (0, 0)
    elif ramp_order >= 1:
        offset, ramp = param_cache[0][0], (param_cache[1][0], param_cache[2][0])
    elif ramp_order == 0:
        offset, ramp = param_cache[0][0], (0, 0)
    else:
        raise ValueError('ramp_order has to be a positive integer or None!')
    # Plot stuff:
    if plot_results:
        if ar_dens is None:
            ar_dens = np.max([1, np.max(dim) // 64])
        axis = magdata_rec.plot_field('Reconstructed Distribution', figsize=(15, 15))
        magdata_rec.plot_quiver(axis=axis, ar_dens=ar_dens, coloring='uniform')
        phasemap.plot_combined('Input Phase')
        phasemap -= fwd_model.ramp(index=0)
        phasemap.plot_combined('Input Phase (ramp corrected)')
        phasemap_rec = pm(magdata_rec)
        title = 'Reconstructed Phase'
        if ramp_order is not None:
            if ramp_order >= 0:
                print('offset:', offset)
                title += ', fitted Offset: {:.2g} [rad]'.format(offset)
            if ramp_order >= 1:
                print('ramp:', ramp)
                title += ', (Fitted Ramp: (u:{:.2g}, v:{:.2g}) [rad/nm]'.format(*ramp)
        phasemap_rec.plot_combined(title)
        diff = (phasemap_rec - phasemap).phase
        diff_name = 'Difference (mean: {:.2g})'.format(diff.mean())
        (phasemap_rec - phasemap).plot_phase(diff_name, sigma_clip=3)
        if ramp_order is not None:
            fwd_model.ramp(0).plot_combined('Fitted Ramp')
    # Return reconstructed magnetisation distribution and cost function:
    return magdata_rec, cost

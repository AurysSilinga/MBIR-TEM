# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Reconstruct a magnetization distributions from a single phase map."""

import logging

import numpy as np

from .. import reconstruction
from ..dataset import DataSet, DataSetCharge
from ..projector import SimpleProjector
from ..regularisator import FirstOrderRegularisator, NoneRegularisator, ZeroOrderRegularisator
from ..kernel import KernelCharge
from ..phasemapper import PhaseMapperCharge
from ..forwardmodel import ForwardModel, ForwardModelCharge
from ..costfunction import Costfunction
from .pm import pm

__all__ = ['reconstruction_2d_from_phasemap']
_log = logging.getLogger(__name__)

# TODO: lam should NOT have a default!!!


def reconstruction_2d_from_phasemap(phasemap, b_0=1, lam=1E-3, max_iter=100, ramp_order=None,
                                    plot_results=False, ar_dens=None, verbose=True):
    """Convenience function for reconstructing a projected distribution from a single phasemap.

    Parameters
    ----------
    phasemap: :class:`~PhaseMap`
        The phasemap which is used for the reconstruction.
    b_0 : float, optional
        The magnetic induction corresponding to a magnetization :math:`M_{0}` in T.
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
        arrows (a number of 2 plots every second arrow). Will be estimated if not provided.
    verbose: bool, optional
        If set to True, information like a progressbar is displayed during reconstruction.
        The default is False.

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
    magdata_rec = reconstruction.optimize_linear(cost, max_iter=max_iter, verbose=verbose)
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
        magdata_rec.plot_quiver_field(note='Reconstructed Distribution',
                                      ar_dens=ar_dens, figsize=(16, 16))
        phasemap_rec = pm(magdata_rec)
        gain = 4 * 2 * np.pi / (np.abs(phasemap_rec.phase).max() + 1E-30)
        gain = round(gain, -int(np.floor(np.log10(abs(gain)))))
        vmin = phasemap_rec.phase.min()
        vmax = phasemap_rec.phase.max()
        phasemap.plot_combined(note='Input Phase', gain=gain)
        phasemap -= fwd_model.ramp(index=0)
        phasemap.plot_combined(note='Input Phase (ramp corrected)', gain=gain, vmin=vmin, vmax=vmax)
        title = 'Reconstructed Phase'
        if ramp_order is not None:
            if ramp_order >= 0:
                print('offset:', offset)
                # title += ', fitted Offset: {:.2g} [rad]'.format(offset)
            if ramp_order >= 1:
                print('ramp:', ramp)
                # title += ', (Fitted Ramp: (u:{:.2g}, v:{:.2g}) [rad/nm]'.format(*ramp)
        phasemap_rec.plot_combined(note=title, gain=gain, vmin=vmin, vmax=vmax)
        diff = (phasemap_rec - phasemap)
        diff_name = 'Difference (RMS: {:.2g} rad)'.format(np.sqrt(np.mean(diff.phase) ** 2))
        diff.plot_phase_with_hist(note=diff_name, sigma_clip=3)
        if ramp_order is not None:
            ramp = fwd_model.ramp(0)
            ramp.plot_phase(note='Fitted Ramp')
    # Return reconstructed magnetisation distribution and cost function:
    return magdata_rec, cost


def reconstruction_2d_charge_from_phasemap(phasemap, max_iter=1000, ramp_order=None, mask=None,
                                           lam=None, electrode_vec=(1E6, 1E6), v_acc=300000, prw=None,
                                           plot_results=False, verbose=True):
    """Convenience function for reconstructing a projected distribution from a single phasemap.

    Parameters
    ----------
    phasemap: :class:`~PhaseMap`
        The phasemap which is used for the reconstruction.
    max_iter : int, optional
        The maximum number of iterations for the optimization.
    ramp_order : int or None (default)
        Polynomial order of the additional phase ramp which will be added to the phase maps.
        All ramp parameters have to be at the end of the input vector and are split automatically.
        Default is None (no ramps are added).
    lam: float,
        The zero order regularisator parameter. 'None' means no regularisator.
    mask: ndarrary,
        Define where situate the reconstructed charges
    electrode_vec : tuple of float (N=2)
        The norm vector of the counter electrode in pixels, (elec_a,elec_b), and the distance to the origin is
        the norm of (elec_a,elec_b).
    v_acc: float
        The accelerating voltage of electrons.
    prw: tuple of 2 int, optional
        A two-component vector describing the displacement of the reference wave to include
        perturbation of this reference by the object itself (via fringing fields), (y, x).
    plot_results: boolean, optional
        If True, the results are plotted after reconstruction.
    verbose: bool, optional
        If set to True, information like a progressbar is displayed during reconstruction.
        The default is False.

    Returns
    -------
    elecdata_rec, cost: :class:`~.ScalarData`, :class:`~.Costfunction`
        The reconstructed magnetisation distribution and the used costfunction.

    """
    _log.debug('Calling reconstruction_2d_charge_from_phasemap')
    # Construct DataSet, Regularisator, ForwardModel and Costfunction:
    dim = (1,) + phasemap.dim_uv
    data = DataSetCharge(phasemap.a, dim, electrode_vec, mask=mask)
    kernel = KernelCharge(phasemap.a, phasemap.dim_uv, electrode_vec=electrode_vec,
                          v_acc=v_acc, prw_vec=prw)
    data.append(phasemap, SimpleProjector(dim), PhaseMapperCharge(kernel))
    data.set_3d_mask()
    # TODO: Rework classes below (ForwardModel, Costfunction)!
    fwd_model = ForwardModelCharge(data, ramp_order)
    if lam is None:
        reg = NoneRegularisator()  # FirstOrderRegularisator(data.mask, lam, add_params=fwd_model.ramp.n)
    else:
        # reg = FirstOrderRegularisator(data.mask, lam=lam, p=2, add_params=fwd_model.ramp.n, factor=1)
        reg = ZeroOrderRegularisator(data.mask, lam=lam, add_params=fwd_model.ramp.n)
    cost = Costfunction(fwd_model, reg)
    # Reconstruct:
    elecdata_rec = reconstruction.optimize_linear_charge(cost, max_iter=max_iter, verbose=verbose)
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
        phasemap_rec = pm(elecdata_rec, mapper='Charge')
        gain = 4 * 2 * np.pi / (np.abs(phasemap_rec.phase).max() + 1E-30)
        gain = round(gain, -int(np.floor(np.log10(abs(gain)))))
        vmin = phasemap_rec.phase.min()
        vmax = phasemap_rec.phase.max()
        phasemap.plot_combined(note='Input Phase', gain=gain)
        phasemap -= fwd_model.ramp(index=0)
        phasemap.plot_combined(note='Input Phase (ramp corrected)', gain=gain, vmin=vmin, vmax=vmax)
        title = 'Reconstructed Phase'
        if ramp_order is not None:
            if ramp_order >= 0:
                print('offset:', offset)
                # title += ', fitted Offset: {:.2g} [rad]'.format(offset)
            if ramp_order >= 1:
                print('ramp:', ramp)
                # title += ', (Fitted Ramp: (u:{:.2g}, v:{:.2g}) [rad/nm]'.format(*ramp)
        phasemap_rec.plot_combined(note=title, gain=gain, vmin=vmin, vmax=vmax)
        diff = (phasemap_rec - phasemap)
        diff_name = 'Difference (RMS: {:.2g} rad)'.format(np.sqrt(np.mean(diff.phase) ** 2))
        diff.plot_phase_with_hist(note=diff_name, sigma_clip=3)
        if ramp_order is not None:
            ramp = fwd_model.ramp(0)
            ramp.plot_phase(note='Fitted Ramp')
    # Return reconstructed charge distribution and cost function:
    return elecdata_rec, cost

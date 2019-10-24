# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""Reconstruct a magnetization distributions from phase maps created from it."""

import logging

import numpy as np

import multiprocessing as mp

from .. import reconstruction
from ..dataset import DataSet, DataSetCharge
from ..projector import XTiltProjector, YTiltProjector
from ..ramp import Ramp
from ..regularisator import FirstOrderRegularisator, NoneRegularisator, ZeroOrderRegularisator
from ..forwardmodel import ForwardModel, DistributedForwardModel, ForwardModelCharge
from ..costfunction import Costfunction
from ..phasemapper import PhaseMapperRDFC, PhaseMapperCharge
from ..kernel import Kernel, KernelCharge

__all__ = ['reconstruction_3d_from_magdata', 'reconstruction_3d_from_elecdata']
_log = logging.getLogger(__name__)


def reconstruction_3d_from_magdata(magdata, b_0=1, lam=1E-3, max_iter=100, ramp_order=1,
                                   angles=np.linspace(-90, 90, num=19), dim_uv=None,
                                   axes=(True, True), noise=0, offset_max=0, ramp_max=0,
                                   use_internal_mask=True, plot_results=False, plot_input=False,
                                   ar_dens=None, multicore=False, verbose=True):
    """Convenience function for reconstructing a projected distribution from a single phasemap.

    Parameters
    ----------
    magdata: :class:`~.VectorData`
        The magnetisation distribution which should be used for the reconstruction.
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
    angles: :class:`~numpy.ndarray` (N=1), optional
        Numpy array determining the angles which should be used for the projectors in x- and
        y-direction. This implicitly sets the number of images per rotation axis. Defaults to a
        range from -90° to 90° degrees, in 10° steps.
    dim_uv: int or None (default)
        Determines if the phasemaps should be padded to a certain size while calculating.
    axes: tuple of booleans (N=2), optional
        Determines if both tilt axes should be calculated. The order is (x, y), both are True by
        default.
    noise: float, optional
        If this is not zero, random gaussian noise with this as a maximum value will be applied
        to all calculated phasemaps. The default is 0. The unit is radians.
    offset_max: float, optional
        if this is not zero, a random offset with this as a maximum value will be applied to all
        calculated phasemaps. The default is 0.
    ramp_max: float, optional
        if this is not zero, a random linear ramp with this as a maximum value will be applied
        to both axes of all calculated phasemaps. The default is 0.
    use_internal_mask: boolean, optional
        If True, the mask from the input magnetization distribution is taken for the
        reconstruction. If False, the mask is calculated via logic backprojection from the 2D-masks
        of the input phasemaps.
    plot_results: boolean, optional
        If True, the results are plotted after reconstruction.
    plot_input:
        If True, the input phasemaps are plotted after reconstruction.
    ar_dens: int, optional
        Number defining the arrow density which is plotted. A higher ar_dens number skips more
        arrows (a number of 2 plots every second arrow). Default is 1.
    multicore: boolean, optional
        Determines if multiprocessing should be used. Default is True. Phasemap calculations
        will be divided onto the separate cores.
    verbose: bool, optional
        If set to True, information like a progressbar is displayed during reconstruction.
        The default is False.

    Returns
    -------
    magdata_rec, cost: :class:`~.VectorData`, :class:`~.Costfunction`
        The reconstructed magnetisation distribution and the used costfunction.

    """
    _log.debug('Calling reconstruction_3d_from_magdata')
    # Construct DataSet:
    dim = magdata.dim
    if ar_dens is None:
        ar_dens = np.max([1, np.max(dim) // 128])
    data = DataSet(magdata.a, magdata.dim, b_0)
    # Construct projectors:
    projectors = []
    # Construct data set and regularisator:
    for angle in angles:
        angle_rad = angle * np.pi / 180
        if axes[0]:
            projectors.append(XTiltProjector(magdata.dim, angle_rad, dim_uv))
        if axes[1]:
            projectors.append(YTiltProjector(magdata.dim, angle_rad, dim_uv))
    # Add pairs of projectors and according phasemaps to the DataSet:
    for projector in projectors:
        mag_proj = projector(magdata)
        phasemap = PhaseMapperRDFC(Kernel(magdata.a, projector.dim_uv, b_0))(mag_proj)
        phasemap.mask = mag_proj.get_mask()[0, ...]
        data.append(phasemap, projector)
    # Add offset and ramp if necessary:
    for i, phasemap in enumerate(data.phasemaps):
        offset = np.random.uniform(-offset_max, offset_max)
        ramp_u = np.random.uniform(-ramp_max, ramp_max)
        ramp_v = np.random.uniform(-ramp_max, ramp_max)
        phasemap += Ramp.create_ramp(phasemap.a, phasemap.dim_uv, (offset, ramp_u, ramp_v))
        data.phasemaps[i] = phasemap
    # Add noise if necessary:
    if noise != 0:  # TODO: write function to add noise after APERTURE!! (ask Florian again)
        for i, phasemap in enumerate(data.phasemaps):
            phasemap.phase += np.random.normal(0, noise, phasemap.dim_uv)
            data.phasemaps[i] = phasemap
    # Construct mask:
    if use_internal_mask:
        data.mask = magdata.get_mask()  # Use perfect mask from magdata!
    else:
        data.set_3d_mask()  # Construct mask from 2D phase masks!
    # Construct regularisator, forward model and costfunction:
    if multicore:
        mp.freeze_support()
        fwd_model = DistributedForwardModel(data, ramp_order=ramp_order, nprocs=mp.cpu_count())
    else:
        fwd_model = ForwardModel(data, ramp_order=ramp_order)
    reg = FirstOrderRegularisator(data.mask, lam, add_params=fwd_model.ramp.n)
    cost = Costfunction(fwd_model, reg)
    # Reconstruct and save:
    magdata_rec = reconstruction.optimize_linear(cost, max_iter=max_iter, verbose=verbose)
    # Finalize ForwardModel (returns workers if multicore):
    fwd_model.finalize()
    # Plot input:
    if plot_input:
        data.plot_phasemaps()
    # Plot results:
    if plot_results:
        data.plot_mask()
        magdata.plot_quiver3d('Original Distribution', ar_dens=ar_dens)
        magdata_rec.plot_quiver3d('Reconstructed Distribution (angle)', ar_dens=ar_dens)
        magdata_rec.plot_quiver3d('Reconstructed Distribution (amplitude)',
                                  ar_dens=ar_dens, coloring='amplitude')
    # Return reconstructed magnetisation distribution and cost function:
    return magdata_rec, cost


def reconstruction_3d_from_elecdata(elecdata, electrode_vec=(1E6, 1E6), max_iter=100, ramp_order=1,
                                    angles=np.linspace(-90, 90, num=19), dim_uv=None, lam=None,
                                    axes=(True, False), noise=0, offset_max=0, ramp_max=0,
                                    confidence=None, use_internal_mask=False, mask_3d_threshold=0.9,
                                    mask=None, plot_results=False, plot_input=False,
                                    multicore=False, verbose=True):
    """Convenience function for reconstructing a projected distribution from a single phasemap.

    Parameters
    ----------1
    elecdata: :class:`~.ScalarData`
        The Charge distribution which should be used for the reconstruction.
        The default is 1.
    electrode_vec : tuple of float (N=2)
        The norm vector of the counter electrode in pixels, (elec_a,elec_b), and the distance to the
        origin is the norm of (elec_a,elec_b).
    max_iter : int, optional
        The maximum number of iterations for the optimization.
    ramp_order : int or None (default)
        Polynomial order of the additional phase ramp which will be added to the phase maps.
        All ramp parameters have to be at the end of the input vector and are split automatically.
        Default is None (no ramps are added).
    angles: :class:`~numpy.ndarray` (N=1), optional
        Numpy array determining the angles which should be used for the projectors in x- and
        y-direction. This implicitly sets the number of images per rotation axis. Defaults to a
        range from -90° to 90° degrees, in 10° steps.
    dim_uv: int or None (default)
        Determines if the phasemaps should be padded to a certain size while calculating.
    lam: float,
        The zero order regularisator parameter. 'None' means no regularisator.
    axes: tuple of booleans (N=2), optional
        Determines if both tilt axes should be calculated. The order is (x, y), both are True by
        default.
    noise: float, optional
        If this is not zero, random gaussian noise with this as a maximum value will be applied
        to all calculated phasemaps. The default is 0. The unit is radians.
    offset_max: float, optional
        if this is not zero, a random offset with this as a maximum value will be applied to all
        calculated phasemaps. The default is 0.
    ramp_max: float, optional
        if this is not zero, a random linear ramp with this as a maximum value will be applied
        to both axes of all calculated phasemaps. The default is 0.
    confidencee: boolean, optional
        if not None, define the trust of the phase image. Here we use the phasemap.mask.
    use_internal_mask: boolean, optional
        If '3D', the mask from the input charge distribution is taken for the
        reconstruction. If '2D', the mask is calculated via logic backprojection from the 2D-masks
        of the input phasemaps. If 'manual', the mask is given by user.
    mask_3d_threshold: float, optional，
        Provide if 'use_internal_mask' is '2D', which determines the mask from phasemaps.
    mask: :class:`~numpy.ndarray` (N=3), provide if `use_internal_mask` set to manual
        A boolean mask which defines the magnetized volume in 3D.
    plot_results: boolean, optional
        If True, the results are plotted after reconstruction.
    plot_input:
        If True, the input phasemaps are plotted after reconstruction.
    multicore: boolean, optional
        Determines if multiprocessing should be used. Default is True. Phasemap calculations
        will be divided onto the separate cores.
    verbose: bool, optional
        If set to True, information like a progressbar is displayed during reconstruction.
        The default is False.

    Returns
    -------
    elecdata_rec, cost: :class:`~.ScalarData`, :class:`~.Costfunction`
        The reconstructed charge distribution and the used costfunction.

    """
    _log.debug('Calling reconstruction_3d_from_elecdata')
    # Construct DataSet:
    data = DataSetCharge(elecdata.a, elecdata.dim, electrode_vec=electrode_vec)
    # Construct projectors:
    projectors = []
    # Construct data set and regularisator:
    for angle in angles:
        angle_rad = angle * np.pi / 180
        if axes[0]:
            projectors.append(XTiltProjector(elecdata.dim, angle_rad, dim_uv))
        if axes[1]:
            projectors.append(YTiltProjector(elecdata.dim, angle_rad, dim_uv))
    # Add pairs of projectors and according phasemaps to the DataSet:
    for projector in projectors:
        elec_proj = projector(elecdata)
        kernel_charge = KernelCharge(elecdata.a, projector.dim_uv, electrode_vec)
        phasemap = PhaseMapperCharge(kernel_charge)(elec_proj)
        phasemap.mask = elec_proj.get_mask()[0, ...]
        if confidence is not None:
            phasemap.confidence = np.logical_not(elec_proj.get_mask()[0, ...])
        data.append(phasemap, projector)
    # Add offset and ramp if necessary:
    for i, phasemap in enumerate(data.phasemaps):
        offset = np.random.uniform(-offset_max, offset_max)
        ramp_u = np.random.uniform(-ramp_max, ramp_max)
        ramp_v = np.random.uniform(-ramp_max, ramp_max)
        phasemap += Ramp.create_ramp(phasemap.a, phasemap.dim_uv, (offset, ramp_u, ramp_v))
        data.phasemaps[i] = phasemap
    # Add noise if necessary:
    if noise != 0:  # TODO: write function to add noise after APERTURE!! (ask Florian again)
        for i, phasemap in enumerate(data.phasemaps):
            phasemap.phase += np.random.normal(0, noise, phasemap.dim_uv)
            data.phasemaps[i] = phasemap
    # Construct mask:
    if use_internal_mask == '3D':
        data.mask = elecdata.get_mask()  # Use perfect mask from elecdata!
    elif use_internal_mask == '2D':
        data.set_3d_mask(threshold=mask_3d_threshold)  # Construct mask from 2D phase masks!
    elif use_internal_mask == 'manual':
        data.mask = mask
    elif not use_internal_mask:
        data.mask = None
    else:
        raise ValueError('use_internal_mask value is wrong')
    # Construct regularisator, forward model and costfunction:
    if multicore:
        mp.freeze_support()
        fwd_model = DistributedForwardModel(data, ramp_order=ramp_order, nprocs=mp.cpu_count())
    else:
        fwd_model = ForwardModelCharge(data, ramp_order=ramp_order)
    if lam is None:
        reg = NoneRegularisator()
        # TODO: or? FirstOrderRegularisator(data.mask, lam, add_params=fwd_model.ramp.n)
    else:
        reg = ZeroOrderRegularisator(lam=lam, add_params=fwd_model.ramp.n)
    # reg = NoneRegularisator()
    cost = Costfunction(fwd_model, reg)
    # Reconstruct and save:
    elecdata_rec = reconstruction.optimize_linear_charge(cost, max_iter=max_iter, verbose=verbose)
    # Finalize ForwardModel (returns workers if multicore):
    fwd_model.finalize()
    # Plot input:
    if plot_input:
        data.plot_phasemaps(symmetric=False)
    # Plot results:
    if plot_results:
        data.plot_mask()
        elecdata_rec.plot_field(cmap='viridis')
    # Return reconstructed charge distribution and cost function:
    return elecdata_rec, cost

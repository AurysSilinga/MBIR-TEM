# -*- coding: utf-8 -*-
"""Reconstruct a magnetization distributions from phase maps created from it."""

import numpy as np

import multiprocessing as mp

from jutil.taketime import TakeTime

from .. import reconstruction
from ..fielddata import VectorData
from ..dataset import DataSet
from ..projector import XTiltProjector, YTiltProjector
from ..ramp import Ramp
from ..regularisator import FirstOrderRegularisator
from ..forwardmodel import ForwardModel, DistributedForwardModel
from ..costfunction import Costfunction


def reconstruction_3d_from_magdata(filename, b_0=1, lam=1E-3, max_iter=100, ramp_order=1,
                                   angles=np.linspace(-90, 90, num=19), dim_uv=None,
                                   axes=(True, True), noise=0, offset_max=0, ramp_max=0,
                                   use_internal_mask=True, plot_results=False, plot_input=False,
                                   ar_dens=None, multicore=True):

    mag_data = VectorData.load_from_hdf5(filename)
    dim = mag_data.dim
    # Load magnetization distribution:
    if ar_dens is None:
        ar_dens = np.max(dim) // 64
    data = DataSet(mag_data.a, mag_data.dim, b_0)
    # Construct projectors:
    projectors = []
    # Construct data set and regularisator:
    for angle in angles:
        angle_rad = angle * np.pi / 180
        if axes[0]:
            projectors.append(XTiltProjector(mag_data.dim, angle_rad, dim_uv))
        if axes[1]:
            projectors.append(YTiltProjector(mag_data.dim, angle_rad, dim_uv))
    data.projectors = projectors
    data.phase_maps = data.create_phase_maps(mag_data)
    # Add projectors and construct according phase maps:
    for i, phase_map in enumerate(data.phase_maps):
        offset = np.random.uniform(-offset_max, offset_max)
        ramp_u = np.random.uniform(-ramp_max, ramp_max)
        ramp_v = np.random.uniform(-ramp_max, ramp_max)
        phase_map += Ramp.create_ramp(phase_map.a, phase_map.dim_uv, (offset, ramp_u, ramp_v))
    # Add noise if necessary:
    if noise != 0:
        for i, phase_map in enumerate(data.phase_maps):
            phase_map.phase += np.random.normal(0, noise, phase_map.dim_uv)
            data.phase_maps[i] = phase_map
    # Construct mask:
    if use_internal_mask:
        data.mask = mag_data.get_mask()  # Use perfect mask from mag_data!
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
    with TakeTime('reconstruction time'):
        mag_data_rec = reconstruction.optimize_linear(cost, max_iter=max_iter)
    # Finalize ForwardModel (returns workers if multicore):
    fwd_model.finalize()
    # Plot input:
    if plot_input:
        data.phase_plots()
    # Plot results:
    if plot_results:
        data.display_mask(ar_dens=ar_dens)
        mag_data.quiver_plot3d('Original Distribution', ar_dens=ar_dens)
        mag_data_rec.quiver_plot3d('Reconstructed Distribution (angle)', ar_dens=ar_dens)
        mag_data_rec.quiver_plot3d('Reconstructed Distribution (amplitude)',
                                   ar_dens=ar_dens, coloring='amplitude')
    # Return reconstructed magnetisation distribution and cost function:
    return mag_data_rec, cost

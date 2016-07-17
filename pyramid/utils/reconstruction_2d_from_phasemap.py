# -*- coding: utf-8 -*-
"""Reconstruct a magnetization distributions from a single phase map."""

import numpy as np

from jutil.taketime import TakeTime

from .. import reconstruction
from ..phasemap import PhaseMap
from ..dataset import DataSet
from ..projector import SimpleProjector
from ..regularisator import FirstOrderRegularisator
from ..forwardmodel import ForwardModel
from ..costfunction import Costfunction
from .pm import pm


def reconstruction_2d_from_phasemap(filename, b_0=1, lam=1E-3, max_iter=100, ramp_order=1,
                                    plot_results=False, ar_dens=None):

    # Load phase map:
    phase_map = PhaseMap.load_from_hdf5(filename)
    # Construct regularisator, forward model and costfunction:
    dim = (1,) + phase_map.dim_uv
    data = DataSet(phase_map.a, dim, b_0)
    data.append(phase_map, SimpleProjector(dim))
    data.set_3d_mask()
    fwd_model = ForwardModel(data, ramp_order)
    reg = FirstOrderRegularisator(data.mask, lam, add_params=fwd_model.ramp.n)
    cost = Costfunction(fwd_model, reg)
    # Reconstruct:
    with TakeTime('reconstruction time'):
        mag_data_rec = reconstruction.optimize_linear(cost, max_iter=max_iter)
        param_cache = cost.fwd_model.ramp.param_cache
    if ramp_order >= 1:
        offset, ramp = param_cache[0][0], (param_cache[1][0], param_cache[2][0])
    elif ramp_order == 0:
        offset, ramp = param_cache[0][0], (0, 0)
    else:
        offset, ramp = 0, (0, 0)
    # Plot stuff:
    if plot_results:
        if ar_dens is None:
            ar_dens = np.max(dim) // 128
        mag_data_rec.quiver_plot('Reconstructed Distribution', ar_dens=ar_dens)
        phase_map.display_combined('Input Phase')
        phase_map -= fwd_model.ramp(index=0)
        phase_map.display_combined('Input Phase (ramp corrected)')
        phase_map_rec = pm(mag_data_rec)
        title = 'Reconstructed Phase'
        if ramp_order >= 0:
            print('offset:', offset)
            title += ', fitted Offset: {:.2g} [rad]'.format(offset)
        if ramp_order >= 1:
            print('ramp:', ramp)
            title += ', (Fitted Ramp: (u:{:.2g}, v:{:.2g}) [rad/nm]'.format(*ramp)
        phase_map_rec.display_combined(title)
        difference = (phase_map_rec.phase - phase_map.phase).mean()
        (phase_map_rec - phase_map).display_phase('Difference (mean: {:.2g})'.format(difference))
        if ramp_order is not None:
            fwd_model.ramp(0).display_combined('Fitted Ramp')
    # Return reconstructed magnetisation distribution and cost function:
    return mag_data_rec, cost

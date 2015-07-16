# -*- coding: utf-8 -*-
"""Reconstruct a magnetization distributions from a single phase map."""


import numpy as np
import pyramid as pr
from jutil.taketime import TakeTime
import logging.config


logging.config.fileConfig(pr.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
phase_name = 'phasemap_dm3_zi_an_elongated_nanorod'
b_0 = 0.6  # in T
lam = 1E-3
max_iter = 100
buffer_pixel = 0
fit_ramps = True
fit_offsets = True
###################################################################################################

# Load phase map:
phase_map = pr.PhaseMap.load_from_netcdf4(phase_name+'.nc')
phase_map.pad((buffer_pixel, buffer_pixel))
dim = (1,) + phase_map.dim_uv

# Construct regularisator, forward model and costfunction:
data = pr.DataSet(phase_map.a, dim, b_0)
data.append(phase_map, pr.SimpleProjector(dim))
data.set_3d_mask()
if fit_ramps:
    add_param_count = 3
elif fit_offsets:
    add_param_count = 1
else:
    add_param_count = None
reg = pr.FirstOrderRegularisator(data.mask, lam, add_params=add_param_count)
fwd_model = pr.ForwardModel(data, fit_offsets=fit_offsets, fit_ramps=fit_ramps)
cost = pr.Costfunction(fwd_model, reg)

# Reconstruct and save:
with TakeTime('reconstruction time'):
    mag_data_rec, add_info = pr.reconstruction.optimize_linear(cost, max_iter=max_iter)
if fit_ramps:
    offset, ramp = add_info[0][0], add_info[1][0]
elif fit_offsets:
    offset = add_info[0][0]
mag_data_buffer = mag_data_rec.copy()
mag_data_rec.crop((0, buffer_pixel, buffer_pixel))
mag_name = '{}_lam={}'.format(phase_name.replace('phasemap', 'magdata_rec'), lam)
mag_data_rec.save_to_netcdf4(mag_name+'.nc')

# Plot stuff:
mag_data_rec.quiver_plot('Reconstructed Distribution', ar_dens=np.ceil(np.max(dim)/128.))
phase_map.crop((buffer_pixel, buffer_pixel))
phase_map.display_combined('Input Phase')
phase_map_rec = pr.pm(mag_data_rec)
phase_map_rec.mask = phase_map.mask
title = 'Reconstructed Phase'
if fit_offsets:
    print 'offset:', offset
    title += ', fitted Offset: {:.2g} [rad]'.format(offset)
if fit_ramps:
    print 'ramp:', ramp
    title += ', (Fitted Ramp: (u:{:.2g}, v:{:.2g}) [rad/nm]'.format(*ramp)
phase_map_rec.display_combined(title)
difference = (phase_map_rec.phase-phase_map.phase).mean()
(phase_map_rec-phase_map).display_phase('Difference (mean: {:.2g})'.format(difference))

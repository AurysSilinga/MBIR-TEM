# -*- coding: utf-8 -*-
"""Reconstruct a magnetization distributions from a single phase map."""


import numpy as np
import pyramid as py
from jutil.taketime import TakeTime
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
phase_name = 'phasemap_bmp_trevor_magnetite_1'
b_0 = 1
lam = 1E-3
max_iter = 100
buffer_pixel = 20
###################################################################################################

# Load phase map:
phase_map = py.PhaseMap.load_from_netcdf4(phase_name+'.nc')
phase_map.pad(buffer_pixel, buffer_pixel)
dim = (1,) + phase_map.dim_uv

# Construct data set and regularisator:
data = py.DataSet(phase_map.a, dim, b_0)
data.append(phase_map, py.SimpleProjector(dim))
data.set_3d_mask()
reg = py.FirstOrderRegularisator(data.mask, lam)

# Reconstruct and save:
with TakeTime('reconstruction time'):
    mag_data_rec, cost = py.reconstruction.optimize_linear(data, reg, max_iter=max_iter)
mag_data_buffer = mag_data_rec.copy()
mag_data_rec.crop([0]*2+[buffer_pixel]*4)
mag_name = '{}_lam={}'.format(phase_name.replace('phasemap', 'magdata_rec'), lam)
mag_data_rec.save_to_netcdf4(mag_name+'.nc')

# Plot stuff:
mag_data_rec.quiver_plot('Reconstructed Distribution', ar_dens=np.ceil(np.max(dim)/64.))
phase_map.crop([buffer_pixel]*4)
phase_map.display_combined('Input Phase')
phase_map_rec = py.pm(mag_data_rec)
phase_map_rec.mask = phase_map.mask
phase_map_rec.display_combined('Reconstructed Phase')
difference = (phase_map_rec.phase-phase_map.phase).mean()
(phase_map_rec-phase_map).display_phase('Difference (mean: {:.3g})'.format(difference))

bp = buffer_pixel
mag_data_buffer.magnitude[:, :, bp:-bp, bp:-bp] = 0
mag_data_buffer.quiver_plot(ar_dens=np.ceil(np.max(dim)/64.))
py.pm(mag_data_buffer).display_phase()

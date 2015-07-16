# -*- coding: utf-8 -*-
"""Reconstruct a magnetization distributions from phase maps created from it."""


import numpy as np
import pyramid as pr
from jutil.taketime import TakeTime
import logging.config


logging.config.fileConfig(pr.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
mag_name = 'magdata_mc_vortex_disc'
dim_uv = None
angles = np.linspace(-90, 90, num=7)
axes = {'x': True, 'y': True}
b_0 = 1
noise = 0.1
lam = 1E-1
max_iter = 100
use_internal_mask = True
offset_max = 2
ramp_max = 0.01
fit_ramps = True
fit_offsets = True
###################################################################################################

# Load magnetization distribution:
mag_data = pr.MagData.load_from_netcdf4(mag_name+'.nc')
dim = mag_data.dim

# Construct data set and regularisator:
data = pr.DataSet(mag_data.a, mag_data.dim, b_0)

# Construct projectors:
projectors = []
for angle in angles:
    angle_rad = angle*np.pi/180
    if axes['x']:
        projectors.append(pr.XTiltProjector(mag_data.dim, angle_rad, dim_uv))
    if axes['y']:
        projectors.append(pr.YTiltProjector(mag_data.dim, angle_rad, dim_uv))

# Add projectors and construct according phase maps:
data.projectors = projectors
data.phase_maps = data.create_phase_maps(mag_data)

for i, phase_map in enumerate(data.phase_maps):
    offset = np.random.uniform(-offset_max, offset_max)
    ramp = np.random.uniform(-ramp_max, ramp_max), np.random.uniform(-ramp_max, ramp_max)
    phase_map.add_ramp(offset, ramp)

# Add noise if necessary:
if noise != 0:
    for i, phase_map in enumerate(data.phase_maps):
        phase_map.phase += np.random.normal(0, noise, phase_map.dim_uv)

        data.phase_maps[i] = phase_map

# Plot input:
for i, phase_map in enumerate(data.phase_maps):
    phase_map.display_phase()

# Construct mask:
if use_internal_mask:
    data.mask = mag_data.get_mask()  # Use perfect mask from mag_data!
else:
    data.set_3d_mask()  # Construct mask from 2D phase masks!

# Construct regularisator, forward model and costfunction:
if fit_ramps:
    add_param_count = 3 * data.count
elif fit_offsets:
    add_param_count = data.count
else:
    add_param_count = None
reg = pr.FirstOrderRegularisator(data.mask, lam, add_params=add_param_count)
fwd_model = pr.ForwardModel(data, fit_offsets=fit_offsets, fit_ramps=fit_ramps)
cost = pr.Costfunction(fwd_model, reg)

# Reconstruct and save:
with TakeTime('reconstruction time'):
    mag_data_rec, add_info = pr.reconstruction.optimize_linear(cost, max_iter=max_iter)
if fit_ramps:
    offset, ramp = add_info[0], add_info[1]
    print 'offsets:', offset
    print 'ramps:', ramp
elif fit_offsets:
    offset = add_info[0]
    print 'offset:', offset
mag_name_rec = mag_name.replace('magdata', 'magdata_rec')
mag_data_rec.save_to_netcdf4(mag_name_rec+'.nc')

# Plot stuff:
data.display_mask(ar_dens=np.ceil(np.max(dim)/64.))
mag_data.quiver_plot3d('Original Distribution', ar_dens=np.ceil(np.max(dim)/64.))
mag_data_rec.quiver_plot3d('Reconstructed Distribution', ar_dens=np.ceil(np.max(dim)/64.))
mag_data_rec.quiver_plot3d('Reconstructed Distribution', ar_dens=np.ceil(np.max(dim)/64.),
                           coloring='amplitude')

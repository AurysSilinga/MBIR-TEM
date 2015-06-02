# -*- coding: utf-8 -*-
"""Reconstruct a magnetization distributions from phase maps created from it."""


import numpy as np
import pyramid as py
from jutil.taketime import TakeTime
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
mag_name = 'magdata_mc_array_sphere_disc_slab'
dim_uv = None
angles = np.linspace(-90, 90, num=7)
axes = {'x': True, 'y': True}
b_0 = 1
noise = 0
lam = 1E-4
max_iter = 100
save_images = False
use_internal_mask = False
###################################################################################################

# Load magnetization distribution:
mag_data = py.MagData.load_from_netcdf4(mag_name+'.nc')
dim = mag_data.dim

# Construct data set and regularisator:
data = py.DataSet(mag_data.a, mag_data.dim, b_0)

# Construct projectors:
projectors = []
for angle in angles:
    angle_rad = angle*np.pi/180
    if axes['x']:
        projectors.append(py.XTiltProjector(mag_data.dim, angle_rad, dim_uv))
    if axes['y']:
        projectors.append(py.YTiltProjector(mag_data.dim, angle_rad, dim_uv))

# Add projectors and construct according phase maps:
data.projectors = projectors
data.phase_maps = data.create_phase_maps(mag_data)

# Add noise if necessary:
if noise != 0:
    for i, phase_map in enumerate(data.phase_maps):
        phase_map += py.PhaseMap(mag_data.a, np.random.normal(0, noise, dim_uv))
        data.phase_maps[i] = phase_map

# Construct mask:
if use_internal_mask:
    data.mask = mag_data.get_mask()  # Use perfect mask from mag_data!
else:
    data.set_3d_mask()  # Construct mask from 2D phase masks!

# Construct regularisator:
reg = py.FirstOrderRegularisator(data.mask, lam)

# Reconstruct and save:
with TakeTime('reconstruction time'):
    mag_data_rec, cost = py.reconstruction.optimize_linear(data, reg, max_iter=max_iter)
mag_name_rec = mag_name.replace('magdata', 'magdata_rec')
mag_data_rec.save_to_netcdf4(mag_name_rec+'.nc')

# Plot stuff:
data.display_mask(ar_dens=np.ceil(np.max(dim)/64.))
mag_data.quiver_plot3d('Original Distribution', ar_dens=np.ceil(np.max(dim)/64.))
mag_data_rec.quiver_plot3d('Reconstructed Distribution', ar_dens=np.ceil(np.max(dim)/64.))

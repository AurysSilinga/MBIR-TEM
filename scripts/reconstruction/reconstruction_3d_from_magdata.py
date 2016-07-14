# -*- coding: utf-8 -*-
"""Reconstruct a magnetization distributions from phase maps created from it."""

import multiprocessing as mp

import matplotlib.pyplot as plt

import numpy as np

import pyramid as pr
from jutil.taketime import TakeTime


###################################################################################################
mag_name = 'magdata_mc_alternating_vortices'
dim_uv = None
angles = np.linspace(-90, 90, num=19)
axes = {'x': True, 'y': True}
b_0 = 1
noise = 0
lam = 1E-5
max_iter = 100
use_internal_mask = True
offset_max = 0
ramp_max = 0
order = 1
show_input = True
ar_dens = 1 # 'auto'
nprocs = mp.cpu_count()  # or 1 for non-multiprocessing
###################################################################################################


if __name__ == '__main__':
    mp.freeze_support()
    # Load magnetization distribution:
    mag_data = pr.VectorData.load_from_hdf5(mag_name + '.hdf5')
    dim = mag_data.dim
    if ar_dens == 'auto':
        ar_dens = np.ceil(np.max(dim) / 64)
    # Construct data set and regularisator:
    data = pr.DataSet(mag_data.a, mag_data.dim, b_0)
    # Construct projectors:
    projectors = []
    for angle in angles:
        angle_rad = angle * np.pi / 180
        if axes['x']:
            projectors.append(pr.XTiltProjector(mag_data.dim, angle_rad, dim_uv))
        if axes['y']:
            projectors.append(pr.YTiltProjector(mag_data.dim, angle_rad, dim_uv))
    # Add projectors and construct according phase maps:
    data.projectors = projectors
    data.phase_maps = data.create_phase_maps(mag_data)
    for i, phase_map in enumerate(data.phase_maps):
        offset = np.random.uniform(-offset_max, offset_max)
        ramp_u = np.random.uniform(-ramp_max, ramp_max)
        ramp_v = np.random.uniform(-ramp_max, ramp_max)
        phase_map += pr.Ramp.create_ramp(phase_map.a, phase_map.dim_uv, (offset, ramp_u, ramp_v))
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
    if nprocs > 1:
        fwd_model = pr.DistributedForwardModel(data, ramp_order=order, nprocs=nprocs)
    else:
        fwd_model = pr.ForwardModel(data, ramp_order=order)
    reg = pr.FirstOrderRegularisator(data.mask, lam, add_params=fwd_model.ramp.n)
    cost = pr.Costfunction(fwd_model, reg)
    # Reconstruct and save:
    with TakeTime('reconstruction time'):
        mag_data_rec = pr.reconstruction.optimize_linear(cost, max_iter=max_iter)
        param_cache = cost.fwd_model.ramp.param_cache
    fwd_model.finalize()
    mag_name_rec = mag_name.replace('magdata', 'magdata_rec')
    mag_data_rec.save_to_hdf5(mag_name_rec + '.hdf5', overwrite=True)
    # Plot stuff:
    data.display_mask(ar_dens=ar_dens)
    mag_data.quiver_plot3d('Original Distribution', ar_dens=ar_dens)
    mag_data_rec.quiver_plot3d('Reconstructed Distribution', ar_dens=ar_dens)
    mag_data_rec.quiver_plot3d('Reconstructed Distribution', ar_dens=ar_dens, coloring='amplitude')
    # Plot input:
    if show_input:
        data.display_phase()
    plt.show()

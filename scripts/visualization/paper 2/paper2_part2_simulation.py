# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:27:39 2015

@author: Jan
"""


import os
import matplotlib.pyplot as plt
import pyramid as py


###################################################################################################
magname_2part_sim = 'magdata_mc_paper2_homog_slab_simulated.nc'
magname_4part_sim = 'magdata_mc_paper2_vortex_slab_simulated.nc'
magname_2part_sim_rec = 'magdata_rec_mc_paper2_homog_slab_simulated_lam=0.0001.nc'
magname_4part_sim_rec = 'magdata_rec_mc_paper2_vortex_slab_simulated_lam=0.0001.nc'

dim = (1, 512, 512)
b_0 = 0.6
lam = 1E-4
ar_dens = 4
###################################################################################################

directory = os.path.join(py.DIR_FILES, 'images', 'paper 2')
if not os.path.exists(directory):
    os.makedirs(directory)

# Original distribution:
mag_data_2part_sim = py.MagData.load_from_netcdf4(magname_2part_sim)
mag_data_4part_sim = py.MagData.load_from_netcdf4(magname_4part_sim)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Magnetic Distributions (simulated data)', fontsize=20)
mag_data_2part_sim.quiver_plot('2 particle array', axis=axes[0], ar_dens=ar_dens)
axes[0].set_aspect('equal')
axes[0].set_xlim(128, 384)
axes[0].set_ylim(128, 384)
axes[0].set_xlabel('x [nm]', fontsize=15)
axes[0].set_ylabel('y [nm]', fontsize=15)
mag_data_4part_sim.quiver_plot('4 particle vortex array', axis=axes[1], ar_dens=ar_dens)
axes[1].set_aspect('equal')
axes[1].set_xlim(128, 384)
axes[1].set_ylim(128, 384)
axes[1].set_xlabel('x [nm]', fontsize=15)
axes[1].set_ylabel('y [nm]', fontsize=15)
plt.savefig(os.path.join(directory, 'Original distributions.png'),
            bbox_inches='tight', dpi=300)

# Original phase maps:
phase_map_2part_sim = py.pm(mag_data_2part_sim)
phase_map_4part_sim = py.pm(mag_data_4part_sim)
axis_p, axis_h = phase_map_2part_sim.display_combined('Simulated phase map (2 particle array)')
axis_p.set_xlabel('x [nm]', fontsize=15)
axis_p.set_ylabel('y [nm]', fontsize=15)
axis_h.set_xlabel('x [nm]', fontsize=15)
axis_h.set_ylabel('y [nm]', fontsize=15)
plt.savefig(os.path.join(directory, 'Original phase map 2 particles.png'),
            bbox_inches='tight', dpi=300)
axis_p, axis_h = phase_map_4part_sim.display_combined('Simulated phase map (4 particle array)')
axis_p.set_xlabel('x [nm]', fontsize=15)
axis_p.set_ylabel('y [nm]', fontsize=15)
axis_h.set_xlabel('x [nm]', fontsize=15)
axis_h.set_ylabel('y [nm]', fontsize=15)
plt.savefig(os.path.join(directory, 'Original phase map 4 particles.png'),
            bbox_inches='tight', dpi=300)

# Reconstructed distribution:
mag_data_2part_sim_rec = py.MagData.load_from_netcdf4(magname_2part_sim_rec)
mag_data_4part_sim_rec = py.MagData.load_from_netcdf4(magname_4part_sim_rec)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Magnetic Distributions (reconstructed, sim.)', fontsize=20)
mag_data_2part_sim_rec.quiver_plot('2 particle array', axis=axes[0], ar_dens=ar_dens)
axes[0].set_aspect('equal')
axes[0].set_xlim(128, 384)
axes[0].set_ylim(128, 384)
axes[0].set_xlabel('x [nm]', fontsize=15)
axes[0].set_ylabel('y [nm]', fontsize=15)
mag_data_4part_sim_rec.quiver_plot('4 particle vortex array', axis=axes[1], ar_dens=ar_dens)
axes[1].set_aspect('equal')
axes[1].set_xlim(128, 384)
axes[1].set_ylim(128, 384)
axes[1].set_xlabel('x [nm]', fontsize=15)
axes[1].set_ylabel('y [nm]', fontsize=15)
plt.savefig(os.path.join(directory, 'Reconstructed distributions (sim).png'),
            bbox_inches='tight', dpi=300)

# Reconstructed phase maps:
phase_map_2part_sim_rec = py.pm(mag_data_2part_sim_rec)
phase_map_4part_sim_rec = py.pm(mag_data_4part_sim_rec)
name = 'Reconstructed phase map (2 particle array)'
axis_p, axis_h = phase_map_2part_sim_rec.display_combined(name)
axis_p.set_xlabel('x [nm]', fontsize=15)
axis_p.set_ylabel('y [nm]', fontsize=15)
axis_h.set_xlabel('x [nm]', fontsize=15)
axis_h.set_ylabel('y [nm]', fontsize=15)
plt.savefig(os.path.join(directory, 'Reconstructed phase map 2 particles (sim).png'),
            bbox_inches='tight', dpi=300)
name = 'Reconstructed phase map (4 particle array)'
axis_p, axis_h = phase_map_4part_sim_rec.display_combined(name)
axis_p.set_xlabel('x [nm]', fontsize=15)
axis_p.set_ylabel('y [nm]', fontsize=15)
axis_h.set_xlabel('x [nm]', fontsize=15)
axis_h.set_ylabel('y [nm]', fontsize=15)
plt.savefig(os.path.join(directory, 'Reconstructed phase map 4 particles (sim).png'),
            bbox_inches='tight', dpi=300)

phase_map_2part_sim_diff = phase_map_2part_sim_rec - phase_map_2part_sim
difference = phase_map_2part_sim_diff.phase.mean()
phase_map_2part_sim_diff.display_phase('Difference (mean: {:.3g})'.format(difference))
plt.savefig(os.path.join(directory, 'Difference phase map 2 particles (sim).png'),
            bbox_inches='tight', dpi=300)

phase_map_4part_sim_diff = phase_map_4part_sim_rec - phase_map_4part_sim
difference = phase_map_4part_sim_diff.phase.mean()
phase_map_4part_sim_diff.display_phase('Difference (mean: {:.3g})'.format(difference))
plt.savefig(os.path.join(directory, 'Difference phase map 4 particles (sim).png'),
            bbox_inches='tight', dpi=300)

#plt.close('all')

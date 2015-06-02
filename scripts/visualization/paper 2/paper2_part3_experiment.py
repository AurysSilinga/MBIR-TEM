# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:27:39 2015

@author: Jan
"""


import os
import matplotlib.pyplot as plt
import pyramid as py


###################################################################################################
phasename_2part_exp = 'phasemap_dm3_zi_an_magnetite_2_particles.nc'
phasename_4part_exp = 'phasemap_dm3_zi_an_magnetite_4_particles.nc'
magname_2part_exp_rec = 'magdata_rec_dm3_zi_an_magnetite_2_particles_lam=0.0001.nc'
magname_4part_exp_rec = 'magdata_rec_dm3_zi_an_magnetite_4_particles_lam=0.0001.nc'

dim = (1, 512, 512)
b_0 = 0.6
lam = 1E-4
ar_dens = 4
###################################################################################################

directory = os.path.join(py.DIR_FILES, 'images', 'paper 2')
if not os.path.exists(directory):
    os.makedirs(directory)

# Original phase maps:
phase_map_2part_exp = py.PhaseMap.load_from_netcdf4(phasename_2part_exp)
phase_map_4part_exp = py.PhaseMap.load_from_netcdf4(phasename_4part_exp)
axis_p, axis_h = phase_map_2part_exp.display_combined('Simulated phase map (2 particle array)')
axis_p.set_xlabel('x [nm]', fontsize=15)
axis_p.set_ylabel('y [nm]', fontsize=15)
axis_h.set_xlabel('x [nm]', fontsize=15)
axis_h.set_ylabel('y [nm]', fontsize=15)
plt.savefig(os.path.join(directory, 'Experimental phase map 2 particles.png'),
            bbox_inches='tight', dpi=300)
axis_p, axis_h = phase_map_4part_exp.display_combined('Simulated phase map (4 particle array)')
axis_p.set_xlabel('x [nm]', fontsize=15)
axis_p.set_ylabel('y [nm]', fontsize=15)
axis_h.set_xlabel('x [nm]', fontsize=15)
axis_h.set_ylabel('y [nm]', fontsize=15)
plt.savefig(os.path.join(directory, 'Experimental phase map 4 particles.png'),
            bbox_inches='tight', dpi=300)

# Reconstructed distribution:
mag_data_2part_exp_rec = py.MagData.load_from_netcdf4(magname_2part_exp_rec)
mag_data_4part_exp_rec = py.MagData.load_from_netcdf4(magname_4part_exp_rec)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Magnetic Distributions (reconstructed, exp.)', fontsize=20)
mag_data_2part_exp_rec.quiver_plot('2 particle array', axis=axes[0], ar_dens=ar_dens)
axes[0].set_aspect('equal')
axes[0].set_xlim(128, 384)
axes[0].set_ylim(128, 384)
axes[0].set_xlabel('x [nm]', fontsize=15)
axes[0].set_ylabel('y [nm]', fontsize=15)
mag_data_4part_exp_rec.quiver_plot('4 particle vortex array', axis=axes[1], ar_dens=ar_dens)
axes[1].set_aspect('equal')
axes[1].set_xlim(128, 384)
axes[1].set_ylim(128, 384)
axes[1].set_xlabel('x [nm]', fontsize=15)
axes[1].set_ylabel('y [nm]', fontsize=15)
plt.savefig(os.path.join(directory, 'Reconstructed distributions (exp).png'),
            bbox_inches='tight', dpi=300)

# Reconstructed phase maps:
phase_map_2part_exp_rec = py.pm(mag_data_2part_exp_rec)
phase_map_4part_exp_rec = py.pm(mag_data_4part_exp_rec)
name = 'Reconstructed phase map (2 particle array)'
axis_p, axis_h = phase_map_2part_exp_rec.display_combined(name)
axis_p.set_xlabel('x [nm]', fontsize=15)
axis_p.set_ylabel('y [nm]', fontsize=15)
axis_h.set_xlabel('x [nm]', fontsize=15)
axis_h.set_ylabel('y [nm]', fontsize=15)
plt.savefig(os.path.join(directory, 'Reconstructed phase map 2 particles (exp).png'),
            bbox_inches='tight', dpi=300)
name = 'Reconstructed phase map (4 particle array)'
axis_p, axis_h = phase_map_4part_exp_rec.display_combined(name)
axis_p.set_xlabel('x [nm]', fontsize=15)
axis_p.set_ylabel('y [nm]', fontsize=15)
axis_h.set_xlabel('x [nm]', fontsize=15)
axis_h.set_ylabel('y [nm]', fontsize=15)
plt.savefig(os.path.join(directory, 'Reconstructed phase map 4 particles (exp).png'),
            bbox_inches='tight', dpi=300)

phase_map_2part_sim_diff = phase_map_2part_exp_rec - phase_map_2part_exp
difference = phase_map_2part_sim_diff.phase.mean()
phase_map_2part_sim_diff.display_phase('Difference (mean: {:.3g})'.format(difference))
plt.savefig(os.path.join(directory, 'Difference phase map 2 particles (exp).png'),
            bbox_inches='tight', dpi=300)

phase_map_4part_sim_diff = phase_map_4part_exp_rec - phase_map_4part_exp
difference = phase_map_4part_sim_diff.phase.mean()
phase_map_4part_sim_diff.display_phase('Difference (mean: {:.3g})'.format(difference))
plt.savefig(os.path.join(directory, 'Difference phase map 4 particles (exp).png'),
            bbox_inches='tight', dpi=300)

#plt.close('all')

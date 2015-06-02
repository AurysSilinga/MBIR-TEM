# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:27:39 2015

@author: Jan
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
magname_2part_sim = 'magdata_mc_paper2_homog_slab_simulated.nc'
magname_4part_sim = 'magdata_mc_paper2_vortex_slab_simulated.nc'
phasename_2part_exp = 'phasemap_dm3_zi_an_magnetite_2_particles.nc'
phasename_4part_exp = 'phasemap_dm3_zi_an_magnetite_4_particles.nc'
magname_2part_rec = 'magdata_rec_dm3_zi_an_magnetite_2_particles_lam=0.0001.nc'
magname_4part_rec = 'magdata_rec_dm3_zi_an_magnetite_4_particles_lam=0.0001.nc'

dim = (1, 512, 512)
b_0 = 0.6
lam = 1E-4
ar_dens = 10
###################################################################################################

directory = os.path.join(py.DIR_FILES, 'images', 'poster PICO 2015')
if not os.path.exists(directory):
    os.makedirs(directory)

# 2 PARTICLE ARRAY: ###############################################################################

# Simulation:
mag_data_2part_sim = py.MagData.load_from_netcdf4(magname_2part_sim)
phase_map_2part_sim = py.pm(mag_data_2part_sim)
axis = phase_map_2part_sim.display_holo(interpolation='bilinear')
mag_data_2part_sim.quiver_plot('2 particle array (simulation)', scale=0.75,
                               axis=axis, color='w', ar_dens=ar_dens)
axis.set_xlim(128, 384)
axis.set_ylim(128, 384)
axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)
plt.savefig(os.path.join(directory, '2 particle array simulation.png'),
            bbox_inches='tight', dpi=300)

# Input phasemap:
phase_map_2part_exp = py.PhaseMap.load_from_netcdf4(phasename_2part_exp)
axis, _ = phase_map_2part_exp.display_phase('Input phase map from experiment', limit=1.6)
axis.set_xlim(128, 384)
axis.set_ylim(128, 384)
axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)
plt.savefig(os.path.join(directory, '2 particle array input phasemap.png'),
            bbox_inches='tight', dpi=300)

# Reconstructed phase map:
mag_data_2part_rec = py.MagData.load_from_netcdf4(magname_2part_rec)
phase_map_2part_rec = py.pm(mag_data_2part_rec)
axis, _ = phase_map_2part_rec.display_phase('Reconstructed phase map', limit=1.6)
axis.set_xlim(128, 384)
axis.set_ylim(128, 384)
axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)
plt.savefig(os.path.join(directory, '2 particle array reconstruced phasemap.png'),
            bbox_inches='tight', dpi=300)

# Reconstructed distribution:
axis = phase_map_2part_rec.display_holo(interpolation='bilinear')
mag_data_2part_rec.quiver_plot('Reconstruced distribution', scale=0.75,
                               axis=axis, color='w', ar_dens=ar_dens)
axis.set_xlim(128, 384)
axis.set_ylim(128, 384)
axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)
plt.savefig(os.path.join(directory, '2 particle array reconstruced distribution.png'),
            bbox_inches='tight', dpi=300)

phase_map_2part_sim_diff = phase_map_2part_rec - phase_map_2part_exp
print 'RMS =', np.sqrt((phase_map_2part_sim_diff.phase**2).mean())


# 4 PARTICLE ARRAY: ###############################################################################

# Simulation:
mag_data_4part_sim = py.MagData.load_from_netcdf4(magname_4part_sim)
phase_map_4part_sim = py.pm(mag_data_4part_sim)
axis = phase_map_4part_sim.display_holo(interpolation='bilinear')
mag_data_4part_sim.quiver_plot('4 particle array (simulation)', scale=0.75,
                               axis=axis, color='w', ar_dens=ar_dens)
axis.set_xlim(128, 384)
axis.set_ylim(128, 384)
axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)
plt.savefig(os.path.join(directory, '4 particle array simulation.png'),
            bbox_inches='tight', dpi=300)

# Input phasemap:
phase_map_4part_exp = py.PhaseMap.load_from_netcdf4(phasename_4part_exp)
axis, _ = phase_map_4part_exp.display_phase('Input phase map from experiment', limit=3.2)
axis.set_xlim(128, 384)
axis.set_ylim(128, 384)
axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)
plt.savefig(os.path.join(directory, '4 particle array input phasemap.png'),
            bbox_inches='tight', dpi=300)

# Reconstructed phase map:
mag_data_4part_rec = py.MagData.load_from_netcdf4(magname_4part_rec)
phase_map_4part_rec = py.pm(mag_data_4part_rec)
axis, _ = phase_map_4part_rec.display_phase('Reconstructed phase map', limit=3.2)
axis.set_xlim(128, 384)
axis.set_ylim(128, 384)
axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)
plt.savefig(os.path.join(directory, '4 particle array reconstruced phasemap.png'),
            bbox_inches='tight', dpi=300)

# Reconstructed distribution:
axis = phase_map_4part_rec.display_holo(interpolation='bilinear')
mag_data_4part_rec.quiver_plot('Reconstruced distribution', scale=0.5,
                               axis=axis, color='w', ar_dens=ar_dens)
axis.set_xlim(128, 384)
axis.set_ylim(128, 384)
axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
axis.tick_params(axis='both', which='major', labelsize=18)
axis.set_xlabel('x [nm]', fontsize=18)
axis.set_ylabel('y [nm]', fontsize=18)
plt.savefig(os.path.join(directory, '4 particle array reconstruced distribution.png'),
            bbox_inches='tight', dpi=300)

phase_map_4part_sim_diff = phase_map_4part_rec - phase_map_4part_exp
print 'RMS =', np.sqrt((phase_map_4part_sim_diff.phase**2).mean())

#plt.close('all')

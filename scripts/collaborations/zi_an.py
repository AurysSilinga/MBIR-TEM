# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 08:41:23 2014

@author: Jan
"""

import os

import numpy as np

from pyDM3reader import DM3lib as dm3

import matplotlib.pyplot as plt

import pyramid
from pyramid.phasemap import PhaseMap
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMConvolve
import pyramid.reconstruction as rc

from time import clock

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
###################################################################################################
PATH = '../../output/zi-an/'
threshold = 1
a = 1.0  # in nm
density = 5
b_0 = 1
inter = 'none'
dim_small = (64, 64)
smoothed_pictures = True
lam = 7.5E-5
order = 1
###################################################################################################

# Read in files:
if smoothed_pictures:
    dm3_2_mag = dm3.DM3(PATH+'Output333_hw512.dm3').image
    dm3_4_mag = dm3.DM3(PATH+'Output334_hw512.dm3').image
    dm3_2_ele = dm3.DM3(PATH+'Output335.dm3').image
    dm3_4_ele = dm3.DM3(PATH+'Output336.dm3').image
else:
    dm3_2_mag = dm3.DM3(PATH+'07_0102mag60_q3_pha_01_sb280_sc512_vf3_med5.dm3').image
    dm3_4_mag = dm3.DM3(PATH+'18a_0102mag_ub140_62k_q3_pha_01_sb180_sc512_vf3_med5.dm3').image
    dm3_2_ele = dm3.DM3(PATH+'07_0102ele60_q3_pha_01_sb280_sc512_vf3_med5.dm3').image
    dm3_4_ele = dm3.DM3(PATH+'18a_0102ele_ub140_62k_q3_pha_01_sb180_sc512_vf3_med5.dm3').image
# Construct phase maps and masks
phase_map_2 = PhaseMap(a, np.array(dm3_2_mag.resize(dim_small))+0.101)
phase_map_2.display_combined(density=density, interpolation=inter)
plt.savefig(PATH+'phase_map_2part.png')
mask_2 = np.expand_dims(np.where(np.array(dm3_2_ele.resize(dim_small)) >= threshold,
                                 True, False), axis=0)
phase_map_4 = PhaseMap(a, np.array(dm3_4_mag.resize(dim_small))-2.546)
phase_map_4.display_combined(density=density, interpolation=inter)
plt.savefig(PATH+'phase_map_4part.png')
mask_4 = np.expand_dims(np.where(np.array(dm3_4_ele.resize(dim_small)) >= threshold,
                                 True, False), axis=0)
# Reconstruct the magnetic distribution:
tic = clock()
mag_data_rec_2 = rc.optimize_simple_leastsq(phase_map_2, mask_2, b_0, lam=lam, order=order)
print '2 particle reconstruction time:', clock() - tic
tic = clock()
mag_data_rec_4 = rc.optimize_simple_leastsq(phase_map_4, mask_4, b_0, lam=lam, order=order)
print '4 particle reconstruction time:', clock() - tic
# Display the reconstructed phase map and holography image:
phase_map_rec_2 = PMConvolve(a, SimpleProjector(mag_data_rec_2.dim), b_0)(mag_data_rec_2)
phase_map_rec_2.display_combined('Reconstr. Distribution', density=density, interpolation=inter)
plt.savefig(PATH+'phase_map_2part_rec.png')
phase_map_rec_4 = PMConvolve(a, SimpleProjector(mag_data_rec_4.dim), b_0)(mag_data_rec_4)
phase_map_rec_4.display_combined('Reconstr. Distribution', density=density, interpolation=inter)
plt.savefig(PATH+'phase_map_4part_rec.png')
# Plot the magnetization:
axis = (mag_data_rec_2*(1/mag_data_rec_2.magnitude.max())).quiver_plot()
#axis.set_xlim(20, 45)
#axis.set_ylim(20, 45)
plt.savefig(PATH+'mag_data_2part.png')
axis = (mag_data_rec_4*(1/mag_data_rec_4.magnitude.max())).quiver_plot()
#axis.set_xlim(20, 45)
#axis.set_ylim(20, 45)
plt.savefig(PATH+'mag_data_4part.png')
# Display the Phase:
phase_diff_2 = phase_map_rec_2-phase_map_2
phase_diff_2.display_phase('Difference')
plt.savefig(PATH+'phase_map_2part_diff.png')
phase_diff_4 = phase_map_rec_4-phase_map_4
phase_diff_4.display_phase('Difference')
plt.savefig(PATH+'phase_map_4part_diff.png')
# Get the average difference from the experimental results:
print 'Average difference (2 cubes):', np.average(phase_diff_2.phase)
print 'Average difference (4 cubes):', np.average(phase_diff_4.phase)
# Plot holographic contour maps with overlayed magnetic distributions:
axis = phase_map_rec_2.display_holo('Magnetization Overlay', density=0.1, interpolation=inter)
(mag_data_rec_2*(1/mag_data_rec_2.magnitude.max())).quiver_plot(axis=axis)
axis = plt.gca()
axis.set_xlim(20, 45)
axis.set_ylim(20, 45)
plt.savefig(PATH+'phase_map_2part_holo.png')
axis = phase_map_rec_4.display_holo('Magnetization Overlay', density=0.1, interpolation=inter)
(mag_data_rec_4*(1/mag_data_rec_4.magnitude.max())).quiver_plot(axis=axis)
axis = plt.gca()
axis.set_xlim(20, 45)
axis.set_ylim(20, 45)
plt.savefig(PATH+'phase_map_4part_holo.png')

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 08:41:23 2014

@author: Jan
"""

import os

import numpy as np

from PIL import Image

from pyDM3reader import DM3lib as dm3

import pyramid
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMConvolve
import pyramid.magcreator as mc
import pyramid.reconstruction as rc

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
###################################################################################################

im_2_ele = Image.open(PATH+'18a_0102ele_ub140_62k_q3_pha_01_sb180_sc512_vf3_med5.jpg').convert('L')
im_2_mag = Image.open(PATH+'18a_0102mag_ub140_62k_q3_pha_01_sb180_sc512_vf3_med5.jpg').convert('L')
im_4_ele = Image.open(PATH+'07_0102ele60_q3_pha_01_sb280_sc512_vf3_med5.jpg').convert('L')
im_4_mag = Image.open(PATH+'07_0102mag60_q3_pha_01_sb280_sc512_vf3_med5.jpg').convert('L')

dm3_2_ele = dm3.DM3(PATH+'07_0102ele60_q3_pha_01_sb280_sc512_vf3_med5.dm3').image
dm3_2_mag = dm3.DM3(PATH+'07_0102mag60_q3_pha_01_sb280_sc512_vf3_med5.dm3').image
dm3_2_neg = dm3.DM3(PATH+'07_0102neg60_q3_pha_01_sb280_sc512_vf3.dm3').image
dm3_2_pos = dm3.DM3(PATH+'07_0102pos60_q3_pha_01_sb280_sc512_vf3.dm3').image
dm3_4_ele = dm3.DM3(PATH+'18a_0102ele_ub140_62k_q3_pha_01_sb180_sc512_vf3_med5.dm3').image
dm3_4_mag = dm3.DM3(PATH+'18a_0102mag_ub140_62k_q3_pha_01_sb180_sc512_vf3_med5.dm3').image
dm3_4_neg = dm3.DM3(PATH+'18a_0102neg_ub140_62k_q3_pha_01_sb180_sc512_vf3_med5.dm3').image
dm3_4_pos = dm3.DM3(PATH+'18a_0102pos_ub140_62k_q3_pha_01_sb180_sc512_vf3_med5.dm3').image

#phase_map_2 = PhaseMap(a, np.array(im_2_mag.resize(dim_small))/255.-0.32197)
phase_map_2 = PhaseMap(a, np.array(dm3_2_mag.resize(dim_small))-0.09147)
phase_map_2.display_combined(density=density, interpolation=inter)
mask_2 = np.expand_dims(np.where(np.array(dm3_2_ele.resize(dim_small)) > threshold, True, False),
                        axis=0)
#mask_2 = np.expand_dims(np.where(np.array(im_2_ele.resize(dim_small)) > threshold, True, False),
#                        axis=0)

#phase_map_4 = PhaseMap(a, np.array(im_4_mag.resize(dim_small))/255.-0.09230)
phase_map_4 = PhaseMap(a, np.array(dm3_4_mag.resize(dim_small))-0.22569)
phase_map_4.display_combined(density=density, interpolation=inter)
mask_4 = np.expand_dims(np.where(np.array(dm3_4_ele.resize(dim_small)) > threshold, True, False),
                        axis=0)
#mask_4 = np.expand_dims(np.where(np.array(im_4_ele.resize(dim_small)) > threshold, True, False),
#                        axis=0)

# Reconstruct the magnetic distribution:
mag_data_rec_2 = rc.optimize_simple_leastsq(phase_map_2, mask_2, b_0)
mag_data_rec_4 = rc.optimize_simple_leastsq(phase_map_4, mask_4, b_0)

# Display the reconstructed phase map and holography image:
phase_map_rec_2 = PMConvolve(a, SimpleProjector(mag_data_rec_2.dim), b_0)(mag_data_rec_2)
phase_map_rec_2.display_combined('Reconstr. Distribution', density=density, interpolation=inter)
phase_map_rec_4 = PMConvolve(a, SimpleProjector(mag_data_rec_4.dim), b_0)(mag_data_rec_4)
phase_map_rec_4.display_combined('Reconstr. Distribution', density=density, interpolation=inter)

(mag_data_rec_2*(1/mag_data_rec_2.magnitude.max())).quiver_plot()
(mag_data_rec_4*(1/mag_data_rec_4.magnitude.max())).quiver_plot()

(phase_map_rec_2-phase_map_2).display_phase('Difference')
(phase_map_rec_4-phase_map_4).display_phase('Difference')

print 'Average difference (2 cubes):', np.average((phase_map_rec_2-phase_map_2).phase)
print 'Average difference (4 cubes):', np.average((phase_map_rec_4-phase_map_4).phase)

mag_data_test_2 = MagData(a, mc.create_mag_dist_homog(mask_4, -0.7+np.pi))
PMConvolve(a, SimpleProjector(mag_data_test_2.dim))(mag_data_test_2).display_phase()

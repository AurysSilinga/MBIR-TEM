# -*- coding: utf-8 -*-
"""Create magnetization distributions from DM3 (Digital Micrograph 3) files."""

import logging.config
import os

import numpy as np
from PIL import Image

import hyperspy.api as hs
import pyramid as pr

import matplotlib.pyplot as plt

logging.config.fileConfig(pr.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
path_mag = 'zi_an_skyrmions_02_38kx_220K_06p27_r_t_magn_x13_y27_w420_h400.dm3'
path_mask = 'zi_an_skyrmions_02_38kx_220K_06p27_r_t_magn_x13_y27_w420_h400_mask_roi.txt'
path_conf = None
filename = 'phasemap_dm3_{}.hdf5'.format(os.path.splitext(path_mag)[0])
print(filename)
a = 1.
dim_uv = None
threshold = 0.5
flip_up_down = True
###################################################################################################

# Load images:
im_mag_hp = hs.load(os.path.join(pr.DIR_FILES, 'dm3', path_mag))
im_mag = Image.fromarray(im_mag_hp.data)
if path_mask is not None:
    # im_mask_hp = hp.load(os.path.join(pr.DIR_FILES, 'dm3', path_mask))
    mask_data = np.genfromtxt(os.path.join(pr.DIR_FILES, 'dm3', path_mask), delimiter=',')
    im_mask = Image.fromarray(mask_data)  # )im_mask_hp.data)
else:
    im_mask = Image.new('F', im_mag.size, 'white')
if path_conf is not None:
    im_conf_hp = hs.load(os.path.join(pr.DIR_FILES, 'dm3', path_conf))
    im_conf = Image.fromarray(im_conf_hp.data)
else:
    im_conf = Image.new('F', im_mag.size, 'white')
if flip_up_down:
    im_mag = im_mag.transpose(Image.FLIP_TOP_BOTTOM)
    im_mask = im_mask.transpose(Image.FLIP_TOP_BOTTOM)
    im_conf = im_conf.transpose(Image.FLIP_TOP_BOTTOM)
if dim_uv is not None:
    im_mag = im_mag.resize(dim_uv)
    im_mask = im_mask.resize(dim_uv)
    im_conf = im_conf.resize(dim_uv)

# Calculate phase, mask and confidence:
phase = np.asarray(im_mag)
mask = np.where(np.asarray(im_mask) >= threshold, True, False)
confidence = np.where(np.asarray(im_conf) >= threshold, 1, 0)

# Create and save PhaseMap object:
phase_map = pr.PhaseMap(a, phase, mask, confidence, unit='rad')
phase_map.save_to_hdf5(os.path.join(pr.DIR_FILES, 'phasemap', filename), overwrite=True)
phase_map.display_combined()
plt.show()
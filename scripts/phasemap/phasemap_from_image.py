# -*- coding: utf-8 -*-
"""Create magnetization distributions from image-files."""

import logging.config
import os

import numpy as np
from PIL import Image

import pyramid as py
import matplotlib.pyplot as plt

logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
path_mag = 'Arnaud_M.tif'
path_mask = 'Arnaud_MIP_mask.tif'
filename = 'phasemap_{}_{}.hdf5'.format(*list(reversed(path_mag.split('.'))))
a = 2  # nm
dim_uv = None
max_phase = 1
threshold = 0.5
offset = 0
flip_up_down = False
###################################################################################################

# Load magnetic phase image:
im_mag = Image.open(os.path.join(py.DIR_FILES, 'images', path_mag)).convert('P')
if flip_up_down:
    im_mag = im_mag.transpose(Image.FLIP_TOP_BOTTOM)
if dim_uv is not None:
    im_mag = im_mag.resize(dim_uv)
phase = np.asarray(im_mag) / 255. * max_phase - offset

# Create mask:
mask = None
if path_mask is not None:
    im_mask = Image.open(os.path.join(py.DIR_FILES, 'images', path_mask)).convert('P')
    if flip_up_down:
        im_mask = im_mask.transpose(Image.FLIP_TOP_BOTTOM)
    if dim_uv is not None:
        im_mask = im_mask.resize(dim_uv)
    mask = np.where(np.asarray(im_mask) / 255. >= threshold, True, False)

# Create and save PhaseMap object:
phase_map = py.PhaseMap(a, phase, mask, confidence=None, unit='rad')
phase_map.save_to_hdf5(os.path.join(py.DIR_FILES, 'phasemap', filename), overwrite=True)
phase_map.display_combined()
plt.show()

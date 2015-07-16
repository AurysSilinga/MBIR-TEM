# -*- coding: utf-8 -*-
"""Create magnetization distributions from image-files."""


import os
import numpy as np
from PIL import Image
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
path_mag = 'Arnaud_M.tif'
path_ele = 'Arnaud_MIP_mask.tif'
filename = 'phasemap_tif_martial_skyrmion.nc'
a = 2
dim_uv = None
max_phase = 1
threshold = 0.5
offset = 0
###################################################################################################

# Load images:
im_mag = Image.open(os.path.join(py.DIR_FILES, 'images', path_mag)).convert('P')
im_ele = Image.open(os.path.join(py.DIR_FILES, 'images', path_ele)).convert('P')
if dim_uv is not None:
    im_mag = im_mag.resize(dim_uv)
    im_ele = im_ele.resize(dim_uv)
# Calculate phase and mask:
phase = np.asarray(im_mag)/255.*max_phase - offset
mask = np.where(np.asarray(im_ele)/255. >= threshold, True, False)

# Create and save PhaseMap object:
phase_map = py.PhaseMap(a, phase, mask, confidence=None, unit='rad')
phase_map.save_to_netcdf4(os.path.join(py.DIR_FILES, 'phasemap', filename))
phase_map.display_combined()

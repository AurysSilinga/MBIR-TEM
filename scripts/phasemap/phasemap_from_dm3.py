# -*- coding: utf-8 -*-
"""Create magnetization distributions from DM3 (Digital Micrograph 3) files."""


import os
import numpy as np
from pyDM3reader import DM3lib as dm3
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
path_mag = 'zi_an_elongated_nanorod.dm3'
path_ele = 'zi_an_elongated_nanorod_mip.dm3'
filename = 'phasemap_dm3_zi_an_elongated_nanorod.nc'
a = 1.
dim_uv = None
threshold = 4.5
###################################################################################################

# Load images:
im_mag = dm3.DM3(os.path.join(py.DIR_FILES, 'dm3', path_mag)).image
im_ele = dm3.DM3(os.path.join(py.DIR_FILES, 'dm3', path_ele)).image
if dim_uv is not None:
    im_mag = im_mag.resize(dim_uv)
    im_ele = im_ele.resize(dim_uv)
# Calculate phase and mask:
phase = np.asarray(im_mag)
mask = np.where(np.asarray(im_ele) >= threshold, True, False)

# Create and save PhaseMap object:
phase_map = py.PhaseMap(a, phase, mask, confidence=None, unit='rad')
phase_map.save_to_netcdf4(os.path.join(py.DIR_FILES, 'phasemap', filename))
phase_map.display_combined()

# -*- coding: utf-8 -*-
"""Create magnetization distributions from DM3 (Digital Micrograph 3) files."""


import os
import numpy as np
import hyperspy.hspy as hp
from PIL import Image
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
path_mag = 'zi_an_magnetite_4_particles_magnetic.dm3'
path_ele = 'zi_an_magnetite_4_particles_electric_smooth.dm3'
filename = 'phasemap_dm3_zi_an_magnetite_4_particles.nc'
a = 1.
dim_uv = (512, 512)
threshold = 0.5
flip_up_down = True
###################################################################################################

# Load images:
im_mag = Image.fromarray(hp.load(os.path.join(py.DIR_FILES, 'dm3', path_mag)).data)
im_ele = Image.fromarray(hp.load(os.path.join(py.DIR_FILES, 'dm3', path_ele)).data)
if flip_up_down:
    im_mag = im_mag.transpose(Image.FLIP_TOP_BOTTOM)
    im_ele = im_ele.transpose(Image.FLIP_TOP_BOTTOM)
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

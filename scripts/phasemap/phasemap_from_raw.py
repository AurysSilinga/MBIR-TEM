# -*- coding: utf-8 -*-
"""Create magnetization distributions from a raw image format."""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import pyramid as pr


###################################################################################################
path_mag = '83-225x148.raw'
path_mask = path_mag
filename = 'skyrmion_cutout_83.hdf5'
im_size = (225, 148)
dim_uv = None
a = 1.
threshold = -1.9
offset = 0.
###################################################################################################

# Load images:
with open(path_mag, 'rb') as raw_file:
    raw_data = raw_file.read()
im_mag = Image.fromstring('F', im_size, raw_data, 'raw')
with open(path_mask, 'rb') as raw_file:
    raw_data = raw_file.read()
im_mask = Image.fromstring('F', im_size, raw_data, 'raw')
if dim_uv is not None:
    im_mag = im_mag.resize(dim_uv)
    im_mask = im_mask.resize(dim_uv)

# Calculate phase and mask:
phase = np.asarray(im_mag) - offset
mask = np.where(np.asarray(im_mask) >= threshold, True, False)

# Create and save PhaseMap object:
phase_map = pr.PhaseMap(a, phase, mask, confidence=None, unit='rad')
phase_map.save_to_hdf5(filename, overwrite=True)
phase_map.display_combined()
plt.show()

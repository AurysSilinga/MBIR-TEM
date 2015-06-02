# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:17:56 2014

@author: Jan
"""

from PIL import Image
import numpy as np

from jutil.taketime import TakeTime
from pyramid import *  # analysis:ignore

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)

# TODO: read in input via txt-file dictionary?

###################################################################################################
a = 1.455  # in nm
gain = 50
b_0 = 1
lam = 1E-4
PATH = '../../output/patrick/'
PHASE = 'Reza_30_uj_tube_M'
MASK = 'Reza_30_uj_tube_maskbygimp'
FORMAT = '.tif'
longFOV = False
longFOV_string = np.where(longFOV, 'longFOV', 'normalFOV')
IMAGENAME = '{}_{}_{}_'.format(MASK, PHASE, longFOV_string)
PHASE_MAX = 0.5  # -10°: 0.92, 30°: 7.68
PHASE_MIN = -0.5  # -10°: -16.85, 30°: -18.89
PHASE_DELTA = PHASE_MAX - PHASE_MIN
###################################################################################################

# Read in files:
im_mask = Image.open(PATH+MASK+FORMAT)
#im_mask.thumbnail((125, 175), Image.ANTIALIAS)
im_phase = Image.open(PATH+PHASE+FORMAT)
#im_phase.thumbnail((125, 125), Image.ANTIALIAS)

mask = np.array(np.array(im_mask)/255, dtype=bool)
dim_uv = mask.shape
phase = (np.array(im_phase)/255.-0.5) * PHASE_DELTA
pad = dim_uv[0] - phase.shape[0]#25
phase_pad = np.zeros(dim_uv)
phase_pad[pad:, :] = phase#[pad:-pad, pad:-pad] = phase

mask = np.expand_dims(mask, axis=0)
dim = mask.shape

phase_map = PhaseMap(a, phase)
phase_map_pad = PhaseMap(a, phase_pad)

data_set = DataSet(a, dim, b_0, mask)
data_set.append(phase_map_pad, SimpleProjector(dim))

# Create Se_inv
if longFOV:
    mask_Se = np.ones(dim_uv)#np.zeros(dim_uv)
    mask_Se[:pad, :] = 0#[pad:-pad, pad:-pad] = 1
    data_set.set_Se_inv_diag_with_masks([mask_Se])

regularisator = FirstOrderRegularisator(mask, lam, p=2)

with TakeTime('reconstruction time'):
    mag_data_rec = rc.optimize_linear(data_set, regularisator=regularisator, max_iter=500)[0]

phase_map_rec_pad = pm(mag_data_rec)
phase_map_rec = PhaseMap(a, phase_map_rec_pad.phase[pad:, :])#[pad:-pad, pad:-pad])
phase_map_diff = phase_map_rec - phase_map

# Display the reconstructed phase map and holography image:
phase_map.display_combined('Input PhaseMap', gain=gain)
plt.savefig(PATH+IMAGENAME+'ORIGINAL.png')
phase_map_pad.display_combined('Input PhaseMap (padded)', gain=gain)
phase_map_rec_pad.display_combined('Reconstr. Distribution (padded)', gain=gain)
plt.savefig(PATH+IMAGENAME+'RECONSTRUCTION_PADDED.png')
phase_map_rec.display_combined('Reconstr. Distribution', gain=gain)
plt.savefig(PATH+IMAGENAME+'RECONSTRUCTION.png')
phase_map_diff.display_combined('Difference')
plt.savefig(PATH+IMAGENAME+'DIFFERENCE.png')
#mag_data_rec.scale_down(4)
mag_data_rec.quiver_plot(log=True, ar_dens=8)
plt.savefig(PATH+IMAGENAME+'MAGNETIZATION_DOWNSCALE4.png')
